# AUTO-ROTO Testing: Quick Start Guide

This guide gets you running your first tests in 5 minutes.

## Installation

### 1. Install pytest and testing dependencies

```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout psutil
```

### 2. Verify installation

```bash
pytest --version
```

You should see: `pytest 7.x.x or higher`

## Running Tests

### Quick start - run all tests

```bash
cd C:/Users/nicho/Documents/auto_roto_scripts/auto_roto_v2/auto_roto
pytest tests/ -v
```

### Run only data validation tests (fastest, ~2 minutes)

```bash
pytest tests/test_data_validation.py -v
```

### Run tests matching a pattern

```bash
# Only unit tests
pytest tests/ -m unit -v

# Only integration tests
pytest tests/ -m integration -v

# Only quality tests
pytest tests/ -m quality -v

# Only tests about alpha validation
pytest tests/ -k "alpha" -v
```

### Run with coverage report

```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

Coverage report will be in `htmlcov/index.html`

### Run tests in parallel (faster)

```bash
pytest tests/ -n auto -v
```

This uses all CPU cores automatically.

### Stop on first failure (faster debugging)

```bash
pytest tests/ -x -v  # Stop on first failure
pytest tests/ --maxfail=3 -v  # Stop after 3 failures
```

## Project Structure

```
auto_roto/
├── pytest.ini                          # Pytest configuration
├── TESTING_STRATEGY.md                 # Complete testing strategy (110+ pages)
├── TESTING_QUICKSTART.md              # This file
├── tests/
│   ├── __init__.py                    # Package init
│   ├── conftest.py                    # Shared fixtures & helpers
│   ├── test_data_validation.py        # 50 tests - PRIORITY 1
│   ├── test_auto_roto.py              # To be created (60 tests)
│   ├── test_depth_refine.py           # To be created (70 tests)
│   ├── test_vitmatte_refine.py        # To be created (60 tests)
│   ├── test_edge_refine.py            # To be created (50 tests)
│   ├── test_temporal_smooth.py        # To be created (70 tests)
│   ├── test_matte_combine.py          # To be created (60 tests)
│   ├── test_full_pipeline_v5.py       # To be created (40 tests)
│   ├── test_integration.py            # To be created (30 tests)
│   ├── test_performance.py            # To be created (20 tests)
│   └── test_data/                     # Test fixtures
│       ├── sample_test.mp4            # (optional) test video
│       └── expected_outputs/
└── [core modules]
    ├── auto_roto.py                   # ~1544 lines
    ├── depth_refine.py                # ~1615 lines
    ├── vitmatte_refine.py             # ~1575 lines
    ├── edge_refine.py                 # ~900 lines
    ├── temporal_smooth.py             # ~1036 lines
    ├── matte_combine.py               # ~1084 lines
    └── full_pipeline_v5.py            # ~743 lines
```

## Understanding Test Results

### Successful test output

```
tests/test_data_validation.py::TestAlphaMapValidation::test_alpha_is_2d_array PASSED
tests/test_data_validation.py::TestAlphaMapValidation::test_alpha_dtype_is_float PASSED
...
============ 50 passed in 2.34s ============
```

### Failed test output

```
FAILED tests/test_data_validation.py::TestAlphaMapValidation::test_alpha_range_within_bounds

AssertionError: Alpha max 1.5 outside [0,1]

tests/test_data_validation.py:35: AssertionError
```

**How to fix:** Look at the line number (35) in the test file and understand what it's checking.

## Writing Your First Test

### Example 1: Simple unit test

```python
# Add to tests/test_example.py

import pytest
import numpy as np

def test_simple_alpha_values():
    """Test that alpha values are in valid range."""
    alpha = np.array([[0.5, 0.8], [0.2, 1.0]], dtype=np.float32)

    # This assertion should pass
    assert 0.0 <= alpha.min() <= 1.0
    assert 0.0 <= alpha.max() <= 1.0

    # This would fail if alpha had invalid values
    assert not np.isnan(alpha).any()
```

### Example 2: Using fixtures

```python
def test_alpha_shape_with_fixture(sample_alpha_512x512):
    """Test that fixture provides correct shape."""
    alpha = sample_alpha_512x512

    assert alpha.shape == (512, 512)
    assert alpha.dtype in [np.float32, np.float64]
```

**Available fixtures** (from `conftest.py`):
- `sample_alpha_512x512` - Random float32 alpha map
- `sample_frame_512x512` - Random uint8 RGB frame
- `sample_depth_map` - Random float32 depth map
- `sample_alpha_sequence` - List of 10 alpha frames
- `temp_dir` - Temporary directory for outputs
- `timer` - Simple timer for performance tests
- See `conftest.py` for complete list

### Example 3: Parametrized test

```python
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_alpha_at_various_sizes(size):
    """Test with multiple input sizes."""
    alpha = np.random.rand(size, size).astype(np.float32)
    assert alpha.shape == (size, size)
```

## Test Organization Best Practices

### Use descriptive test names

```python
# Good
def test_alpha_range_validation_rejects_values_above_1():
    pass

# Bad
def test_alpha_1():
    pass
```

### Use docstrings to explain tests

```python
def test_temporal_smoothing_reduces_flicker(self, flickering_sequence):
    """
    Temporal smoothing should reduce frame-to-frame deltas.

    This test verifies that the smoothing pipeline reduces
    the maximum pixel difference between adjacent frames,
    which manifests as flickering in the output.
    """
    pass
```

### Group related tests in classes

```python
class TestAlphaValidation:
    """All tests related to alpha map validation."""

    def test_alpha_is_2d(self):
        pass

    def test_alpha_dtype_float(self):
        pass

    def test_alpha_range_0_to_1(self):
        pass
```

### Mark tests appropriately

```python
@pytest.mark.unit  # Fast, isolated test
def test_simple_calculation():
    pass

@pytest.mark.integration  # Uses multiple modules
def test_full_pipeline():
    pass

@pytest.mark.slow  # Takes > 10 seconds
def test_processing_long_sequence():
    pass

@pytest.mark.gpu  # Requires CUDA/GPU
def test_model_inference():
    pass
```

## Common Testing Patterns

### Testing expected errors

```python
def test_invalid_input_raises_error():
    """Function should reject invalid input."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

### Testing with approximate equality

```python
def test_floating_point_calculation():
    """Test that matches within tolerance."""
    result = calculate_something()
    expected = 0.5

    np.testing.assert_almost_equal(result, expected, decimal=5)
```

### Testing array shapes and values

```python
def test_output_shape_and_values(sample_alpha):
    """Test both shape and value properties."""
    result = process_alpha(sample_alpha)

    assert result.shape == sample_alpha.shape
    np.testing.assert_array_less(result, 1.0)
    np.testing.assert_array_less(0.0, result)
```

## Debugging Failed Tests

### 1. Run with verbose output

```bash
pytest tests/test_data_validation.py::TestAlphaMapValidation::test_alpha_range_within_bounds -vv
```

### 2. Print debug information

```python
def test_debug_example(sample_alpha_512x512):
    """Debug: print values during test."""
    alpha = sample_alpha_512x512

    print(f"Shape: {alpha.shape}")
    print(f"Min: {alpha.min()}, Max: {alpha.max()}")
    print(f"Dtype: {alpha.dtype}")

    assert alpha.max() <= 1.0
```

Run with `pytest -s` to see print statements:

```bash
pytest tests/test_data_validation.py::TestAlphaMapValidation -vv -s
```

### 3. Drop into debugger on failure

```python
def test_with_debugger(sample_alpha_512x512):
    """Drop into debugger if assertion fails."""
    alpha = sample_alpha_512x512

    if alpha.max() > 1.0:
        import pdb; pdb.set_trace()  # Debugger will start here
```

### 4. Use pytest fixtures to inspect state

```python
@pytest.fixture
def debug_output(temp_dir):
    """Save debug output to disk for inspection."""
    def _save(data, name):
        path = temp_dir / f"{name}.npy"
        np.save(path, data)
        print(f"Saved to {path}")
    return _save

def test_with_debug(sample_alpha_512x512, debug_output):
    """Save intermediate results for analysis."""
    result = process_alpha(sample_alpha_512x512)
    debug_output(result, "processed_alpha")
    assert result.shape == sample_alpha_512x512.shape
```

## Performance Testing

### Time a test

```bash
pytest tests/ -v --durations=10
```

Shows the 10 slowest tests.

### Profile memory usage

```python
def test_memory_stability(memory_tracker):
    """Test that memory doesn't leak."""
    memory_tracker.start()

    for i in range(1000):
        alpha = np.random.rand(512, 512)
        memory_tracker.sample()

    print(f"Memory growth: {memory_tracker.growth_mb}MB ({memory_tracker.growth_pct}%)")
    assert memory_tracker.growth_pct < 20, "Memory growth too high"
```

## Coverage Reports

### Generate HTML coverage report

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

This shows which lines of code are tested and which are not.

### View coverage in terminal

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

Shows untested lines directly in terminal.

## CI/CD Integration

Tests run automatically on:
- Every push to `master` branch
- Every pull request

See `.github/workflows/tests.yml` for configuration.

## Next Steps

1. **Run the existing tests:**
   ```bash
   pytest tests/test_data_validation.py -v
   ```

2. **Create new test files** following the templates provided

3. **Build toward 70% coverage** with priority on:
   - Data validation (shape, dtype, range)
   - Temporal consistency (no flicker)
   - Memory stability (no leaks)
   - Output quality (edges, details)

4. **Set up CI/CD** to run tests automatically

## Useful Links

- [Pytest Documentation](https://docs.pytest.org/)
- [Numpy Testing](https://numpy.org/doc/stable/reference/testing.html)
- [Parametrize Tests](https://docs.pytest.org/en/stable/example/parametrize.html)
- [Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Markers](https://docs.pytest.org/en/stable/example/markers.html)

## FAQ

**Q: How long does the full test suite take?**
A: ~5-10 minutes depending on hardware. Unit tests take ~2 minutes, integration tests 5-20 minutes.

**Q: Can I skip slow tests?**
A: Yes, use `pytest -m "not slow"` to skip tests marked with `@pytest.mark.slow`

**Q: How do I test GPU functionality?**
A: Tests marked `@pytest.mark.gpu` require CUDA. Skip with `pytest -m "not gpu"` if GPU unavailable.

**Q: What if a test is flaky?**
A: Mark it with `@pytest.mark.flaky` and investigate. See pytest-flaky plugin.

**Q: How do I add test data?**
A: Place files in `tests/test_data/` and reference with the `test_data_dir` fixture.

---

**Happy testing!** Report issues or questions about tests to the AUTO-ROTO team.
