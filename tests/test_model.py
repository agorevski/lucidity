# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Tests for model module"""

from heretic.model import AbliterationParameters


class TestAbliterationParameters:
    """Tests for AbliterationParameters dataclass"""

    def test_abliteration_parameters_creation(self):
        """Test creating AbliterationParameters with all fields.

        Verifies that an AbliterationParameters instance can be created
        with all required fields and that the values are correctly stored.
        """
        params = AbliterationParameters(
            max_weight=1.2,
            max_weight_position=10.0,
            min_weight=0.3,
            min_weight_distance=5.0,
        )
        assert params.max_weight == 1.2
        assert params.max_weight_position == 10.0
        assert params.min_weight == 0.3
        assert params.min_weight_distance == 5.0

    def test_abliteration_parameters_types(self):
        """Test that parameters accept float values.

        Verifies that all AbliterationParameters fields are stored as
        float instances when float values are provided.
        """
        params = AbliterationParameters(
            max_weight=1.0,
            max_weight_position=8.5,
            min_weight=0.5,
            min_weight_distance=4.2,
        )
        assert isinstance(params.max_weight, float)
        assert isinstance(params.max_weight_position, float)
        assert isinstance(params.min_weight, float)
        assert isinstance(params.min_weight_distance, float)

    def test_abliteration_parameters_integer_conversion(self):
        """Test that integer values are accepted and work correctly.

        Verifies that AbliterationParameters accepts integer values
        and stores them correctly for use in calculations.
        """
        params = AbliterationParameters(
            max_weight=1,
            max_weight_position=10,
            min_weight=0,
            min_weight_distance=5,
        )
        # Integers should work fine for the calculations
        assert params.max_weight == 1
        assert params.max_weight_position == 10

    def test_abliteration_parameters_realistic_values(self):
        """Test with realistic abliteration parameter values.

        Verifies that AbliterationParameters works correctly with
        values that fall within the realistic ranges used in the
        main.py objective function.
        """
        # Based on the ranges in main.py's objective function
        params = AbliterationParameters(
            max_weight=1.15,  # Between 0.8 and 1.5
            max_weight_position=15.3,  # Between 0.6 * n_layers and n_layers
            min_weight=0.23,  # Fraction of max_weight
            min_weight_distance=7.2,  # Between 1.0 and 0.6 * n_layers
        )
        assert 0.8 <= params.max_weight <= 1.5
        assert params.min_weight < params.max_weight
        assert params.min_weight_distance > 0


class TestModelGetChat:
    """Tests for Model.get_chat method"""

    def test_get_chat_format(self, mock_settings):
        """Test that get_chat returns correct chat format.

        Verifies that get_chat returns a list with two message dicts
        containing the system prompt and user prompt in the correct format.

        Args:
            mock_settings: Pytest fixture providing mock settings object.
        """
        from unittest.mock import patch
        from heretic.model import Model
        
        with patch('heretic.model.AutoTokenizer'):
            with patch('heretic.model.AutoModelForCausalLM'):
                # Create model without triggering full initialization
                model = Model.__new__(Model)
                model.settings = mock_settings
                
                chat = model.get_chat("Test prompt")
                
                assert len(chat) == 2
                assert chat[0]["role"] == "system"
                assert chat[0]["content"] == mock_settings.system_prompt
                assert chat[1]["role"] == "user"
                assert chat[1]["content"] == "Test prompt"

    def test_get_chat_with_custom_prompt(self, mock_settings):
        """Test get_chat with different prompts.

        Verifies that get_chat correctly handles different user prompts
        while maintaining the same system message across calls.

        Args:
            mock_settings: Pytest fixture providing mock settings object.
        """
        from unittest.mock import patch
        from heretic.model import Model
        
        with patch('heretic.model.AutoTokenizer'):
            with patch('heretic.model.AutoModelForCausalLM'):
                model = Model.__new__(Model)
                model.settings = mock_settings
                
                chat1 = model.get_chat("First prompt")
                chat2 = model.get_chat("Second prompt")
                
                assert chat1[1]["content"] == "First prompt"
                assert chat2[1]["content"] == "Second prompt"
                # System message should be the same
                assert chat1[0] == chat2[0]

    def test_get_chat_preserves_system_prompt(self, mock_settings):
        """Test that system prompt from settings is used.

        Verifies that the system prompt from the settings object is
        correctly included in the chat message list.

        Args:
            mock_settings: Pytest fixture providing mock settings object.
        """
        from unittest.mock import patch
        from heretic.model import Model
        
        with patch('heretic.model.AutoTokenizer'):
            with patch('heretic.model.AutoModelForCausalLM'):
                mock_settings.system_prompt = "Custom system prompt"
                model = Model.__new__(Model)
                model.settings = mock_settings
                
                chat = model.get_chat("Test")
                
                assert chat[0]["content"] == "Custom system prompt"
