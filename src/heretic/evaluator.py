# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .utils import load_prompts, print


class Evaluator:
    """Evaluates model behavior on good and bad prompts.

    Measures KL divergence on good prompts and refusal rates on bad prompts
    to assess model quality during training.

    Attributes:
        settings: Configuration settings for the evaluator.
        model: The model to evaluate.
        good_prompts: List of prompts for measuring KL divergence.
        base_logprobs: Initial log probabilities for good prompts.
        bad_prompts: List of prompts for measuring refusal rates.
        base_refusals: Initial refusal count before training.
    """

    def __init__(self, settings: Settings, model: Model):
        """Initializes the Evaluator with settings and model.

        Args:
            settings: Configuration settings containing prompt datasets and
                refusal markers.
            model: The model to evaluate.
        """
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        """Determines if a response is a refusal.

        Args:
            response: The model's response text to check.

        Returns:
            True if the response contains any refusal marker, False otherwise.
        """
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        """Counts the number of refusals for bad prompts.

        Returns:
            The count of responses that are classified as refusals.
        """
        responses = self.model.get_responses_batched(self.bad_prompts)
        refusals = [response for response in responses if self.is_refusal(response)]
        return len(refusals)

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        """Computes the evaluation score for the current model state.

        Calculates KL divergence between current and base log probabilities
        on good prompts, and counts refusals on bad prompts.

        Returns:
            A tuple containing:
                - score: A tuple of (normalized KL divergence, normalized refusals).
                - kl_divergence: The raw KL divergence value.
                - refusals: The raw refusal count.
        """
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_refusals),
        )

        return score, kl_divergence, refusals
