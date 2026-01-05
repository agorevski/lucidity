# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Tests for configuration module"""

import pytest
from pydantic import ValidationError

from heretic.config import DatasetSpecification, Settings


class TestDatasetSpecification:
    """Tests for DatasetSpecification model"""

    def test_valid_dataset_specification(self):
        """Test creating a valid DatasetSpecification.

        Creates a DatasetSpecification with valid dataset, split, and column
        values and verifies all fields are correctly assigned.
        """
        spec = DatasetSpecification(
            dataset="test/dataset",
            split="train[:100]",
            column="text"
        )
        assert spec.dataset == "test/dataset"
        assert spec.split == "train[:100]"
        assert spec.column == "text"

    def test_dataset_specification_missing_fields(self):
        """Test that missing required fields raise validation error.

        Verifies that creating a DatasetSpecification without required
        split and column fields raises a ValidationError.

        Raises:
            ValidationError: When required fields are missing.
        """
        with pytest.raises(ValidationError):
            DatasetSpecification(dataset="test/dataset")


class TestSettings:
    """Tests for Settings model"""

    def test_default_settings(self, monkeypatch):
        """Test creating Settings with minimal required fields.

        Verifies that Settings can be created with only the required model
        field and that all default values are correctly applied.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert settings.model == "test-model"
        assert settings.batch_size == 0  # auto
        assert settings.max_response_length == 100
        assert settings.system_prompt == "You are a helpful assistant."
        assert settings.kl_divergence_scale == 1.0
        assert settings.n_trials == 200

    def test_settings_with_custom_values(self, monkeypatch):
        """Test creating Settings with custom values.

        Verifies that Settings correctly accepts and stores custom values
        for model, batch_size, max_response_length, n_trials, and system_prompt.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(
            model="custom-model",
            batch_size=16,
            max_response_length=200,
            n_trials=100,
            system_prompt="Custom prompt",
        )
        assert settings.model == "custom-model"
        assert settings.batch_size == 16
        assert settings.max_response_length == 200
        assert settings.n_trials == 100
        assert settings.system_prompt == "Custom prompt"

    def test_settings_refusal_markers_default(self, monkeypatch):
        """Test that default refusal markers are present.

        Verifies that the Settings model includes expected default refusal
        markers such as 'sorry', 'i can't', 'i cannot', 'illegal', and 'harmful'.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert "sorry" in settings.refusal_markers
        assert "i can't" in settings.refusal_markers
        assert "i cannot" in settings.refusal_markers
        assert "illegal" in settings.refusal_markers
        assert "harmful" in settings.refusal_markers

    def test_settings_custom_refusal_markers(self, monkeypatch):
        """Test setting custom refusal markers.

        Verifies that custom refusal markers can be provided and are
        correctly stored in the Settings model.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        custom_markers = ["nope", "no way", "absolutely not"]
        settings = Settings(
            model="test-model",
            refusal_markers=custom_markers
        )
        assert settings.refusal_markers == custom_markers

    def test_settings_dtype_list(self, monkeypatch):
        """Test that dtype list has expected defaults.

        Verifies that the default dtypes list includes 'auto', 'float16',
        and 'float32' options.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert "auto" in settings.dtypes
        assert "float16" in settings.dtypes
        assert "float32" in settings.dtypes

    def test_settings_device_map_default(self, monkeypatch):
        """Test device map default value.

        Verifies that the default device_map value is set to 'auto'.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert settings.device_map == "auto"

    def test_settings_auto_save_defaults(self, monkeypatch):
        """Test auto-save related defaults.

        Verifies that auto_save, output_dir, auto_upload_to_hf, and hf_private
        have their expected default values.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert settings.auto_save is False
        assert settings.output_dir == "./outputs"
        assert settings.auto_upload_to_hf is False
        assert settings.hf_private is False

    def test_settings_evaluate_model_optional(self, monkeypatch):
        """Test that evaluate_model is optional.

        Verifies that evaluate_model defaults to None when not provided
        and can be set to a custom value when specified.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert settings.evaluate_model is None
        
        settings = Settings(
            model="test-model",
            evaluate_model="eval-model"
        )
        assert settings.evaluate_model == "eval-model"

    def test_settings_optimization_params(self, monkeypatch):
        """Test optimization-related parameters.

        Verifies that optimization parameters n_trials, n_startup_trials,
        and kl_divergence_scale have their expected default values.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert settings.n_trials == 200
        assert settings.n_startup_trials == 60
        assert settings.kl_divergence_scale == 1.0

    def test_settings_batch_size_auto(self, monkeypatch):
        """Test that batch_size of 0 means auto.

        Verifies that setting batch_size to 0 is accepted and represents
        automatic batch size determination.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model", batch_size=0)
        assert settings.batch_size == 0
        
    def test_settings_max_batch_size(self, monkeypatch):
        """Test max_batch_size default.

        Verifies that the default max_batch_size value is 128.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        assert settings.max_batch_size == 128

    def test_settings_dataset_specifications(self, monkeypatch):
        """Test that dataset specifications have defaults.

        Verifies that default dataset specifications for good_prompts,
        bad_prompts, good_evaluation_prompts, and bad_evaluation_prompts
        are correctly configured with expected dataset names, splits, and columns.

        Args:
            monkeypatch: Pytest fixture for patching sys.argv.
        """
        monkeypatch.setattr('sys.argv', ['pytest'])
        settings = Settings(model="test-model")
        
        assert settings.good_prompts.dataset == "mlabonne/harmless_alpaca"
        assert settings.good_prompts.split == "train[:400]"
        assert settings.good_prompts.column == "text"
        
        assert settings.bad_prompts.dataset == "mlabonne/harmful_behaviors"
        assert settings.bad_prompts.split == "train[:400]"
        assert settings.bad_prompts.column == "text"
        
        assert settings.good_evaluation_prompts.dataset == "mlabonne/harmless_alpaca"
        assert settings.good_evaluation_prompts.split == "test[:100]"
        
        assert settings.bad_evaluation_prompts.dataset == "mlabonne/harmful_behaviors"
        assert settings.bad_evaluation_prompts.split == "test[:100]"
