# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
from .utils import batchify, empty_cache, print


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    def __init__(self, settings: Settings):
        """Initialize the Model with the given settings.

        Loads and configures a transformer model for causal language modeling.
        Attempts to load the model with different dtypes until one succeeds.

        Args:
            settings: Configuration settings including model name, dtypes,
                device map, and trust_remote_code options.

        Raises:
            Exception: If the model fails to load with all configured dtypes.
        """
        self.settings = settings

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = None
        self.trusted_models = {settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[settings.evaluate_model] = settings.trust_remote_code

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    trust_remote_code=self.trusted_models.get(settings.model),
                )

                # If we reach this point and the model requires trust_remote_code,
                # the user must have confirmed it.
                if self.trusted_models.get(settings.model) is None:
                    self.trusted_models[settings.model] = True

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(["Test"], max_new_tokens=1)
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            print("[green]Ok[/]")
            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(matrices)}[/] matrices per layer"
            )

    def reload_model(self):
        """Reload the model from disk.

        Purges the existing model from memory and reloads it with the same
        dtype. Useful for resetting the model to its original state after
        abliteration or other in-place modifications.
        """
        dtype = self.model.dtype

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            trust_remote_code=self.trusted_models.get(self.settings.model),
        )

        if self.trusted_models.get(self.settings.model) is None:
            self.trusted_models[self.settings.model] = True

    def get_layers(self) -> ModuleList:
        """Get the transformer layers from the model.

        Handles both multimodal models (which have a nested language_model)
        and text-only models.

        Returns:
            The ModuleList containing the transformer layers.
        """
        # Most multimodal models.
        with suppress(Exception):
            return self.model.model.language_model.layers

        # Text-only models.
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        """Get the weight matrices for a specific transformer layer.

        Extracts abliterable weight matrices from the given layer, including
        attention out-projection and MLP down-projection matrices. Supports
        various model architectures including dense models and MoE models.

        Args:
            layer_index: The index of the layer to extract matrices from.

        Returns:
            A dictionary mapping component names (e.g., 'attn.o_proj',
            'mlp.down_proj') to lists of weight tensors.
        """
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            # Handle Triton tensors (e.g., from MXFP4 quantization) by extracting
            # the underlying PyTorch tensor via the .data attribute.
            if hasattr(matrix, "data") and torch.is_tensor(matrix.data):
                matrix = matrix.data
            
            assert torch.is_tensor(matrix)

            if component not in matrices:
                matrices[component] = []

            matrices[component].append(matrix)

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        # gpt-oss MoE.
        with suppress(Exception):
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear.weight)

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear.weight)

        # We need at least one MLP down-projection.
        assert matrices["mlp.down_proj"]

        return matrices

    def get_abliterable_components(self) -> list[str]:
        """Get the names of components that can be abliterated.

        Returns:
            A list of component names (e.g., ['attn.o_proj', 'mlp.down_proj']).
        """
        return list(self.get_layer_matrices(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        """Apply abliteration to the model's weight matrices.

        Orthogonalizes weight matrices against the refusal direction to remove
        the model's tendency to refuse certain prompts. The abliteration weight
        varies by layer based on the provided parameters.

        Args:
            refusal_directions: Tensor of refusal directions per layer, with
                shape (num_layers + 1, hidden_size). The first element is for
                embeddings.
            direction_index: If provided, interpolates between adjacent refusal
                directions at this (possibly fractional) layer index. If None,
                uses per-layer refusal directions.
            parameters: Dictionary mapping component names to their abliteration
                parameters (max_weight, position, min_weight, distance).
        """
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, matrices in self.get_layer_matrices(layer_index).items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                # Projects any right-multiplied vector(s) onto the subspace
                # spanned by the refusal direction.
                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)

                for matrix in matrices:
                    # Ensure projector is on the same device as the matrix for multi-GPU support.
                    device_projector = projector.to(matrix.device)
                    # In-place subtraction is safe as we're not using Autograd.
                    matrix.sub_(weight * (device_projector @ matrix))

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        """Create a chat conversation from a user prompt.

        Args:
            prompt: The user's input message.

        Returns:
            A list of message dictionaries with system and user roles.
        """
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        """Generate model outputs for a batch of prompts.

        Applies the chat template to each prompt and runs generation with
        greedy decoding for deterministic outputs.

        Args:
            prompts: List of user prompts to generate responses for.
            **kwargs: Additional arguments passed to the model's generate method
                (e.g., max_new_tokens, output_hidden_states).

        Returns:
            A tuple containing:
                - The tokenized inputs (BatchEncoding).
                - The generated outputs (token IDs or GenerateOutput).
        """
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        return inputs, self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )

    def get_responses(self, prompts: list[str]) -> list[str]:
        """Get text responses for a batch of prompts.

        Args:
            prompts: List of user prompts to generate responses for.

        Returns:
            List of generated response strings (excluding the input prompts).
        """
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        # Return only the newly generated part.
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def get_responses_batched(self, prompts: list[str]) -> list[str]:
        """Get text responses for prompts, processing in batches.

        Splits prompts into batches according to settings.batch_size to
        manage memory usage.

        Args:
            prompts: List of user prompts to generate responses for.

        Returns:
            List of generated response strings for all prompts.
        """
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(batch):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        """Get residual stream vectors for a batch of prompts.

        Generates one token and extracts the hidden states at the final
        position for each prompt and layer.

        Args:
            prompts: List of user prompts to extract residuals from.

        Returns:
            Tensor of shape (num_prompts, num_layers + 1, hidden_size)
            containing the residual vectors, upcast to float32.
        """
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Hidden states for the first (only) generated token.
        hidden_states = outputs.hidden_states[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        """Get residual stream vectors for prompts, processing in batches.

        Args:
            prompts: List of user prompts to extract residuals from.

        Returns:
            Tensor of shape (num_prompts, num_layers + 1, hidden_size)
            containing concatenated residual vectors from all batches.
        """
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    def get_logprobs(self, prompts: list[str]) -> Tensor:
        """Get log probability distributions over the vocabulary for prompts.

        Generates one token and returns the log-softmax of the logits at that
        position. Uses log probabilities for numerical stability when computing
        KL divergence.

        Args:
            prompts: List of user prompts to get log probabilities for.

        Returns:
            Tensor of shape (num_prompts, vocab_size) containing log
            probabilities over the vocabulary for each prompt.
        """
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Logits for the first (only) generated token.
        logits = outputs.scores[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str]) -> Tensor:
        """Get log probability distributions for prompts, processing in batches.

        Args:
            prompts: List of user prompts to get log probabilities for.

        Returns:
            Tensor of shape (num_prompts, vocab_size) containing concatenated
            log probabilities from all batches.
        """
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        """Stream a chat response to the console and return it.

        Generates a response for the given chat conversation, streaming
        tokens to stdout as they are generated.

        Args:
            chat: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            The complete generated response as a string.
        """
        chat_prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
