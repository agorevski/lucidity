# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Pytest configuration and shared fixtures"""

import pytest
from unittest.mock import MagicMock, Mock
import torch


@pytest.fixture
def mock_settings(monkeypatch):
    """Create a mock Settings object with default values"""
    from heretic.config import Settings, DatasetSpecification
    
    # Prevent Settings from parsing CLI arguments during tests
    monkeypatch.setattr('sys.argv', ['pytest'])
    
    return Settings(
        model="test-model",
        batch_size=4,
        max_response_length=100,
        refusal_markers=[
            "sorry",
            "i can't",
            "i cannot",
            "i won't",
            "i will not",
            "i'm unable",
            "i am unable",
            "illegal",
            "harmful",
            "violat",
        ],
        system_prompt="You are a helpful assistant.",
        good_prompts=DatasetSpecification(
            dataset="test/good",
            split="train[:10]",
            column="text"
        ),
        bad_prompts=DatasetSpecification(
            dataset="test/bad",
            split="train[:10]",
            column="text"
        ),
        good_evaluation_prompts=DatasetSpecification(
            dataset="test/good",
            split="test[:5]",
            column="text"
        ),
        bad_evaluation_prompts=DatasetSpecification(
            dataset="test/bad",
            split="test[:5]",
            column="text"
        ),
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.apply_chat_template = Mock(return_value=["Test prompt"])
    tokenizer.batch_decode = Mock(return_value=["Test response"])
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock transformer model"""
    model = MagicMock()
    model.dtype = torch.float32
    model.device = "cpu"
    
    # Mock model.model.layers
    mock_layers = MagicMock()
    mock_layers.__len__ = Mock(return_value=12)
    model.model.layers = mock_layers
    
    return model


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing"""
    return [
        "What is the capital of France?",
        "How do I make a sandwich?",
        "Tell me a joke.",
    ]


@pytest.fixture
def sample_refusal_responses():
    """Sample responses that should be detected as refusals"""
    return [
        "I'm sorry, but I can't help with that.",
        "I cannot assist with that request.",
        "I won't provide that information.",
        "I'm unable to help with that.",
        "As an AI assistant, I cannot do that.",
        "That would violate ethical boundaries.",
    ]


@pytest.fixture
def sample_non_refusal_responses():
    """Sample responses that should NOT be detected as refusals"""
    return [
        "Here's how you can do that...",
        "The answer is 42.",
        "Sure! I'd be happy to help.",
        "Let me explain that to you.",
    ]
