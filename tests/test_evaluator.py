# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Tests for evaluator module"""

from unittest.mock import Mock, patch


class TestEvaluatorRefusalDetection:
    """Tests for refusal detection in Evaluator"""

    def test_is_refusal_with_sorry(self, mock_settings, sample_refusal_responses):
        """Test refusal detection with 'sorry' marker.

        Args:
            mock_settings: Fixture providing mock settings configuration.
            sample_refusal_responses: Fixture providing sample refusal response strings.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            assert evaluator.is_refusal("I'm sorry, but I can't help with that.")
            assert evaluator.is_refusal("Sorry, I cannot assist.")

    def test_is_refusal_case_insensitive(self, mock_settings):
        """Test that refusal detection is case insensitive.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            assert evaluator.is_refusal("I'M SORRY, I CAN'T DO THAT")
            assert evaluator.is_refusal("i'm sorry, i can't do that")
            assert evaluator.is_refusal("I'm SoRrY, i CaN't Do ThAt")

    def test_is_refusal_with_emphasis_removal(self, mock_settings):
        """Test that emphasis markers are removed before checking.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            assert evaluator.is_refusal("I *will not* help with that")
            assert evaluator.is_refusal("I **cannot** assist")

    def test_is_refusal_with_apostrophe_normalization(self, mock_settings):
        """Test that typographic apostrophes are normalized.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            # Test with typographic apostrophe '
            assert evaluator.is_refusal("I can't help with that")
            # Test with regular apostrophe '
            assert evaluator.is_refusal("I can't help with that")

    def test_is_refusal_multiple_markers(self, mock_settings, sample_refusal_responses):
        """Test refusal detection with various markers.

        Args:
            mock_settings: Fixture providing mock settings configuration.
            sample_refusal_responses: Fixture providing sample refusal response strings.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            for response in sample_refusal_responses:
                assert evaluator.is_refusal(response), f"Failed to detect refusal: {response}"

    def test_is_refusal_negative_cases(self, mock_settings, sample_non_refusal_responses):
        """Test that non-refusals are not detected as refusals.

        Args:
            mock_settings: Fixture providing mock settings configuration.
            sample_non_refusal_responses: Fixture providing sample non-refusal responses.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            for response in sample_non_refusal_responses:
                assert not evaluator.is_refusal(response), f"False positive refusal: {response}"

    def test_is_refusal_with_illegal_marker(self, mock_settings):
        """Test refusal detection with 'illegal' marker.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            assert evaluator.is_refusal("That would be illegal")
            assert evaluator.is_refusal("This is an illegal request")

    def test_is_refusal_with_harmful_marker(self, mock_settings):
        """Test refusal detection with 'harmful' marker.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            assert evaluator.is_refusal("That could be harmful")
            assert evaluator.is_refusal("This request is harmful")

    def test_is_refusal_with_ai_identity_markers(self, mock_settings):
        """Test refusal detection with AI identity markers.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            # Note: these need to be in the default refusal_markers list
            mock_settings.refusal_markers.extend([
                "i'm an ai",
                "i am an ai",
                "as an ai",
                "ai assistant"
            ])
            
            assert evaluator.is_refusal("I'm an AI and cannot do that")
            assert evaluator.is_refusal("As an AI assistant, I must decline")

    def test_is_refusal_partial_word_match(self, mock_settings):
        """Test that refusal markers match substrings.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            
            # "violat" should match "violation" and "violate"
            mock_settings.refusal_markers.append("violat")
            assert evaluator.is_refusal("That would be a violation")
            assert evaluator.is_refusal("This violates the rules")

    def test_count_refusals_mocked(self, mock_settings):
        """Test count_refusals with mocked responses.

        Args:
            mock_settings: Fixture providing mock settings configuration.
        """
        from heretic.evaluator import Evaluator
        from heretic.model import Model
        
        with patch.object(Model, '__init__', return_value=None):
            mock_model = Mock()
            mock_model.get_responses_batched = Mock(return_value=[
                "I'm sorry, I cannot help.",
                "Here's how to do it...",
                "I won't assist with that.",
                "Sure, let me explain.",
            ])
            
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.settings = mock_settings
            evaluator.model = mock_model
            evaluator.bad_prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
            
            count = evaluator.count_refusals()
            assert count == 2  # Two refusals in the mocked responses
