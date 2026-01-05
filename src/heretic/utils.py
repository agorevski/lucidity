# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
import getpass
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypeVar

import questionary
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from questionary import Choice
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def is_notebook() -> bool:
    """Detect if the code is running in a Jupyter notebook environment.

    Checks for specific environment variables (Colab, Kaggle) and IPython shell
    types to determine if running in a notebook context. This is necessary because
    when running as a subprocess (e.g. !heretic), get_ipython() might not be
    available or might not reflect the notebook environment.

    Returns:
        bool: True if running in a notebook environment, False otherwise.
    """
    if os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True

    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False

        shell_name = shell.__class__.__name__
        if shell_name in ["ZMQInteractiveShell", "Shell"]:
            return True

        if "google.colab" in str(shell.__class__):
            return True

        return False
    except (ImportError, NameError, AttributeError):
        return False


def prompt_select(message: str, choices: list[Any], style=None) -> Any:
    """Display a selection prompt and return the user's choice.

    Adapts to notebook or terminal environments, providing numbered options
    in notebooks and an interactive selection menu in terminals.

    Args:
        message: The prompt message to display to the user.
        choices: A list of choices to present. Can be plain values or
            questionary.Choice objects.
        style: Optional questionary style for terminal display.

    Returns:
        Any: The selected choice value.
    """
    if is_notebook():
        print()
        print(message)
        real_choices = []
        for i, choice in enumerate(choices, 1):
            if isinstance(choice, Choice):
                print(f"[{i}] {choice.title}")
                real_choices.append(choice.value)
            else:
                print(f"[{i}] {choice}")
                real_choices.append(choice)

        while True:
            try:
                selection = input("Enter number: ")
                idx = int(selection) - 1
                if 0 <= idx < len(real_choices):
                    return real_choices[idx]
                print(
                    f"[red]Please enter a number between 1 and {len(real_choices)}[/]"
                )
            except ValueError:
                print("[red]Invalid input. Please enter a number.[/]")
    else:
        return questionary.select(message, choices=choices, style=style).ask()


def prompt_text(
    message: str,
    default: str = "",
    unsafe: bool = False,
    qmark: str = "?",
) -> str:
    """Display a text input prompt and return the user's input.

    Adapts to notebook or terminal environments, providing simple input()
    in notebooks and questionary text input in terminals.

    Args:
        message: The prompt message to display to the user.
        default: Default value to use if the user provides no input.
        unsafe: If True, use unsafe_ask() which doesn't catch KeyboardInterrupt.
        qmark: The question mark character to display before the message.

    Returns:
        str: The user's input text, or the default value if empty.
    """
    if is_notebook():
        print()
        prompt_msg = f"{message} [{default}]: " if default else f"{message}: "
        result = input(prompt_msg)
        return result if result else default
    else:
        # For text input, we might need unsafe_ask if requested
        q = questionary.text(message, default=default, qmark=qmark)
        if unsafe:
            return q.unsafe_ask()
        return q.ask()


def prompt_path(message: str, default: str = "", only_directories: bool = False) -> str:
    """Display a file/directory path input prompt and return the user's input.

    Adapts to notebook or terminal environments, providing simple input()
    in notebooks and questionary path input with autocomplete in terminals.

    Args:
        message: The prompt message to display to the user.
        default: Default path value to use if the user provides no input.
        only_directories: If True, only allow directory paths (terminal only).

    Returns:
        str: The user's input path, or the default value if empty.
    """
    if is_notebook():
        print()
        prompt_msg = f"{message} [{default}]: " if default else f"{message}: "
        result = input(prompt_msg)
        return result if result else default
    else:
        return questionary.path(
            message, default=default, only_directories=only_directories
        ).ask()


def prompt_password(message: str) -> str:
    """Display a password input prompt with hidden input.

    Adapts to notebook or terminal environments, using getpass in notebooks
    and questionary password input in terminals.

    Args:
        message: The prompt message to display to the user.

    Returns:
        str: The user's password input.
    """
    if is_notebook():
        print()
        return getpass.getpass(message)
    else:
        return questionary.password(message).ask()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Converts seconds to hours/minutes/seconds format, showing only the
    most significant units needed.

    Args:
        seconds: The duration in seconds to format.

    Returns:
        str: A formatted duration string (e.g., "2h 30m", "5m 10s", "45s").
    """
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def load_prompts(specification: DatasetSpecification) -> list[str]:
    """Load prompts from a dataset based on the given specification.

    Supports loading from local directories (including datasets saved with
    datasets.save_to_disk), HuggingFace repositories, and handles split
    specifications.

    Args:
        specification: A DatasetSpecification containing the dataset path,
            split specification, and column name to extract prompts from.

    Returns:
        list[str]: A list of prompt strings extracted from the specified
            dataset column.
    """
    path = specification.dataset
    split_str = specification.split
    if os.path.isdir(path):
        if Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            # Parse the split instructions.
            ri = ReadInstruction.from_spec(split_str)
            # Associate the split with its number of examples (lines).
            split_name = str(dataset.split)
            name2len = {split_name: len(dataset)}
            # Convert the instructions to absolute indices and select the first one.
            abs_i = ri.to_absolute(name2len)[0]
            # Get the dataset by applying the indices.
            dataset = dataset[abs_i.from_ : abs_i.to]
        else:
            # Path is a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    else:
        # Probably a repository path; let load_dataset figure it out.
        dataset = load_dataset(path, split=split_str)

    return list(dataset[specification.column])


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into batches of a specified size.

    Args:
        items: The list of items to split into batches.
        batch_size: The maximum number of items per batch.

    Returns:
        list[list[T]]: A list of batches, where each batch is a list of items.
            The last batch may contain fewer items than batch_size.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    """Clear GPU memory cache and run garbage collection.

    Detects the available accelerator backend (CUDA, XPU, MLU, SDAA, MUSA, or MPS)
    and empties its cache. Runs garbage collection before and after clearing the
    cache to avoid OOM errors (gc.collect() is not idempotent).

    See https://github.com/p-e-w/heretic/pull/17 for details.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    """Extract and format trial parameters for display.

    Extracts the direction_index and component parameters from an Optuna trial's
    user attributes and formats them as strings.

    Args:
        trial: An Optuna Trial object containing user_attrs with direction_index
            and parameters.

    Returns:
        dict[str, str]: A dictionary mapping parameter names to their formatted
            string values.
    """
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in asdict(parameters).items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    """Generate the introductory section of a README for an abliterated model.

    Creates a formatted markdown string containing model information, abliteration
    parameters, and performance metrics comparing the modified model to the original.

    Args:
        settings: The Settings object containing the model name.
        trial: An Optuna Trial object with abliteration parameters and metrics.
        base_refusals: The number of refusals from the original model.
        bad_prompts: The list of prompts used for refusal testing.

    Returns:
        str: A formatted markdown string for the README introduction.
    """
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
