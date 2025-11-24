# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Tests for utility functions"""

from unittest.mock import Mock

from heretic.model import AbliterationParameters
from heretic.utils import batchify, format_duration, get_trial_parameters


class TestBatchify:
    """Tests for batchify function"""

    def test_batchify_exact_batches(self):
        """Test batchifying when items divide evenly into batches"""
        items = list(range(12))
        batches = batchify(items, 4)
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2, 3]
        assert batches[1] == [4, 5, 6, 7]
        assert batches[2] == [8, 9, 10, 11]

    def test_batchify_uneven_batches(self):
        """Test batchifying when items don't divide evenly"""
        items = list(range(10))
        batches = batchify(items, 3)
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_batchify_single_batch(self):
        """Test when batch size is larger than items"""
        items = [1, 2, 3]
        batches = batchify(items, 10)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_batchify_batch_size_one(self):
        """Test with batch size of 1"""
        items = [1, 2, 3]
        batches = batchify(items, 1)
        assert len(batches) == 3
        assert all(len(batch) == 1 for batch in batches)

    def test_batchify_empty_list(self):
        """Test batchifying an empty list"""
        items = []
        batches = batchify(items, 5)
        assert len(batches) == 0

    def test_batchify_preserves_order(self):
        """Test that batchify preserves item order"""
        items = ['a', 'b', 'c', 'd', 'e']
        batches = batchify(items, 2)
        flattened = [item for batch in batches for item in batch]
        assert flattened == items


class TestFormatDuration:
    """Tests for format_duration function"""

    def test_format_seconds_only(self):
        """Test formatting durations under a minute"""
        assert format_duration(0) == "0s"
        assert format_duration(15) == "15s"
        assert format_duration(45.7) == "46s"
        assert format_duration(59) == "59s"

    def test_format_minutes_and_seconds(self):
        """Test formatting durations with minutes"""
        assert format_duration(60) == "1m 0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3599) == "59m 59s"

    def test_format_hours_and_minutes(self):
        """Test formatting durations with hours"""
        assert format_duration(3600) == "1h 0m"
        assert format_duration(3660) == "1h 1m"
        assert format_duration(7200) == "2h 0m"
        assert format_duration(5400) == "1h 30m"

    def test_format_large_durations(self):
        """Test formatting very large durations"""
        assert format_duration(36000) == "10h 0m"
        assert format_duration(86400) == "24h 0m"

    def test_format_rounding(self):
        """Test that seconds are rounded correctly"""
        assert format_duration(60.4) == "1m 0s"
        assert format_duration(60.6) == "1m 1s"
        assert format_duration(59.4) == "59s"


class TestGetTrialParameters:
    """Tests for get_trial_parameters function"""

    def test_get_trial_parameters_with_direction_index(self):
        """Test extracting trial parameters with direction index"""
        mock_trial = Mock()
        mock_trial.user_attrs = {
            "direction_index": 5.75,
            "parameters": {
                "attn.o_proj": AbliterationParameters(
                    max_weight=1.2,
                    max_weight_position=10.5,
                    min_weight=0.3,
                    min_weight_distance=8.0,
                )
            }
        }
        
        params = get_trial_parameters(mock_trial)
        
        assert params["direction_index"] == "5.75"
        assert params["attn.o_proj.max_weight"] == "1.20"
        assert params["attn.o_proj.max_weight_position"] == "10.50"
        assert params["attn.o_proj.min_weight"] == "0.30"
        assert params["attn.o_proj.min_weight_distance"] == "8.00"

    def test_get_trial_parameters_per_layer(self):
        """Test extracting trial parameters with per-layer direction"""
        mock_trial = Mock()
        mock_trial.user_attrs = {
            "direction_index": None,
            "parameters": {
                "mlp.down_proj": AbliterationParameters(
                    max_weight=1.0,
                    max_weight_position=12.0,
                    min_weight=0.5,
                    min_weight_distance=6.0,
                )
            }
        }
        
        params = get_trial_parameters(mock_trial)
        
        assert params["direction_index"] == "per layer"
        assert params["mlp.down_proj.max_weight"] == "1.00"

    def test_get_trial_parameters_multiple_components(self):
        """Test extracting parameters for multiple components"""
        mock_trial = Mock()
        mock_trial.user_attrs = {
            "direction_index": 7.0,
            "parameters": {
                "attn.o_proj": AbliterationParameters(
                    max_weight=1.1,
                    max_weight_position=9.0,
                    min_weight=0.2,
                    min_weight_distance=5.0,
                ),
                "mlp.down_proj": AbliterationParameters(
                    max_weight=1.3,
                    max_weight_position=11.0,
                    min_weight=0.4,
                    min_weight_distance=7.0,
                )
            }
        }
        
        params = get_trial_parameters(mock_trial)
        
        assert "attn.o_proj.max_weight" in params
        assert "mlp.down_proj.max_weight" in params
        assert len(params) == 9  # 1 direction_index + 4 params * 2 components
