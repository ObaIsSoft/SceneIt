# Testing Guide

This document explains how to run tests for the Visual Search application.

## Setup

1. Install test dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have extracted video frames (required for some tests):
   ```bash
   python app.py --extract
   ```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only API tests
pytest -m api

# Run only integration tests
pytest -m integration
```

### Run Specific Test Files
```bash
pytest test_app.py
```

### Run Specific Test Classes or Functions
```bash
# Run all tests in TestSearchEngine class
pytest test_app.py::TestSearchEngine

# Run a specific test function
pytest test_app.py::TestSearchEngine::test_get_embedding
```

### Verbose Output
```bash
pytest -v
```

### See Print Statements
```bash
pytest -s
```

### Coverage Report
```bash
# Install coverage first
pip install pytest-cov

# Run with coverage
pytest --cov=app --cov-report=html
```

## Test Categories

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Fast execution

### API Tests
- Test Flask endpoints
- Validate request/response formats
- Check error handling

### Integration Tests
- Test complete workflows
- Use real components
- May be slower

## CI/CD Integration

To run tests in a CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Install dependencies
  run: pip install -r requirements.txt

- name: Run tests
  run: pytest --junitxml=test-results.xml

- name: Upload results
  uses: actions/upload-artifact@v2
  with:
    name: test-results
    path: test-results.xml
```

## Writing New Tests

Follow these patterns:

```python
import pytest

class TestYourFeature:
    """Test description."""
    
    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        expected = "something"
        
        # Act
        result = your_function()
        
        # Assert
        assert result == expected
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    yield create_sample_data()
    # Cleanup if needed
```

## Common Issues

### "No database loaded" Errors
Some tests require the database to be extracted first:
```bash
python app.py --extract
```

### Model Download on First Run
First test run may be slow due to model downloads. Subsequent runs will be faster.

### Temporary Files
Tests create temporary files that are automatically cleaned up. If tests fail, you may need to manually clean `/tmp` directory.
