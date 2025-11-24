# Heretic Unit Tests

This directory contains high-level unit tests for the Heretic project.

## Test Structure

- `conftest.py` - Shared pytest fixtures and configuration
- `test_config.py` - Tests for configuration and settings
- `test_utils.py` - Tests for utility functions (batchify, format_duration, etc.)
- `test_evaluator.py` - Tests for refusal detection logic
- `test_model.py` - Tests for model parameters and chat formatting

## Running Tests

### Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=src/heretic --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run specific test
pytest tests/test_config.py::TestSettings::test_default_settings -v
```

### In CI/CD

Tests are automatically run on every push and pull request via GitHub Actions. See `.github/workflows/ci.yml` for the configuration.

## Test Philosophy

These are **high-level unit tests** that:

- Do NOT require downloading actual LLM models
- Do NOT require GPU resources
- Use mocks/fixtures for heavy dependencies (transformers, torch models)
- Focus on testing business logic, data structures, and utility functions
- Run quickly and can be executed in CI/CD pipelines

## Coverage

Current test coverage focuses on:
- ✅ Configuration validation (DatasetSpecification, Settings)
- ✅ Utility functions (batchify, format_duration, get_trial_parameters)
- ✅ Refusal detection logic
- ✅ Abliteration parameters
- ✅ Chat message formatting

## Future Improvements

- Add integration tests that test complete workflows (with model mocks)
- Add performance benchmarks for critical functions
- Add property-based testing with hypothesis
- Increase coverage to 80%+
