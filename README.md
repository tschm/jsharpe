# jsharpe

[![PyPI version](https://badge.fury.io/py/jsharpe.svg)](https://badge.fury.io/py/jsharpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://tschm.github.io/jsharpe/coverage-badge.svg)](https://tschm.github.io/jsharpe/reports/html-coverage/index.html)
[![Downloads](https://static.pepy.tech/personalized-badge/jsharpe?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/jsharpe)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/jsharpe/badge)](https://www.codefactor.io/repository/github/tschm/jsharpe)

**A Python library for rigorous Sharpe ratio analysis and statistical testing.**

## Overview

jsharpe provides comprehensive tools for evaluating trading strategies through the lens of statistical significance. Based on the research of Marcos Lopez de Prado, this library goes beyond simple Sharpe ratio calculations to answer the critical question: *Is this strategy's performance statistically significant, or could it be due to chance?*

### Key Features

- **Probabilistic Sharpe Ratio (PSR)** - Transform Sharpe ratios into probabilities that account for estimation uncertainty
- **Non-Gaussian Returns** - Correct for skewness and excess kurtosis in return distributions
- **Autocorrelation Adjustment** - Handle serial correlation in returns
- **Multiple Testing Corrections** - Control False Discovery Rate (FDR) and Family-Wise Error Rate (FWER) when testing multiple strategies
- **Minimum Track Record Length** - Determine how long you need to observe a strategy for statistical significance
- **Portfolio Optimization** - Minimum variance portfolio weights for correlated assets

## Installation

Install jsharpe from PyPI:

```bash
pip install jsharpe
```

## Quick Start

### End-to-End Worked Example

Start from a raw return series and answer two questions in one pass: *is this
Sharpe ratio real?* (PSR) and *would it survive screening many strategies?*
(FDR-controlled cutoff).

```python
import numpy as np

from jsharpe import control_for_FDR, probabilistic_sharpe_ratio

# Monthly excess returns of a candidate strategy (36 observations)
returns = np.array([
    0.021, -0.014, 0.038, 0.006, -0.022, 0.041,
    0.013, 0.029, -0.008, 0.034, 0.017, -0.011,
    0.025, 0.009, 0.031, -0.019, 0.022, 0.014,
    0.037, -0.006, 0.028, 0.011, 0.019, 0.033,
    0.007, 0.024, -0.013, 0.036, 0.015, 0.027,
    0.004, 0.032, -0.017, 0.023, 0.018, 0.030,
])

# 1. Observed Sharpe ratio of the sample
sr = returns.mean() / returns.std(ddof=1)

# 2. Probabilistic Sharpe Ratio: P[true SR > 0] given the sample
psr = probabilistic_sharpe_ratio(SR=sr, SR0=0, T=len(returns))

# 3. FDR-controlled critical Sharpe ratio when screening many strategies at q=25%
_alpha, _beta, sr_cutoff, _q_hat = control_for_FDR(q=0.25, SR0=0, SR1=sr, p_H1=0.05, T=len(returns))

print(f"Observed SR:   {sr:.3f}")
print(f"PSR (SR > 0):  {psr:.3f}")
print(f"FDR SR cutoff: {sr_cutoff:.3f}")
print(f"Survives FDR:  {sr > sr_cutoff}")
```

```result
Observed SR:   0.815
PSR (SR > 0):  1.000
FDR SR cutoff: 0.352
Survives FDR:  True
```

The strategy clears both bars: its PSR is effectively 1.0 (the true Sharpe ratio
is almost certainly positive) and its observed Sharpe ratio of 0.815 comfortably
exceeds the 0.352 threshold required to control the false-discovery rate at 25%.

### Basic Probabilistic Sharpe Ratio

```python
from jsharpe import probabilistic_sharpe_ratio

# Observed Sharpe ratio: 0.456 (e.g., 3.6% return / 7.9% volatility)
sr = 0.036 / 0.079

# Compute PSR with 24 monthly observations
# Testing against SR0=0 (no skill)
psr = probabilistic_sharpe_ratio(SR=sr, SR0=0, T=24)
print(f"PSR: {psr:.3f}")  # Output: PSR: 0.987
```

The PSR of 0.987 means there's a 98.7% probability that the true Sharpe ratio exceeds zero.

### Accounting for Non-Gaussian Returns

Real returns often exhibit negative skewness and excess kurtosis (fat tails):

```python
from jsharpe import probabilistic_sharpe_ratio

sr = 0.036 / 0.079

# Include skewness and kurtosis estimates
psr = probabilistic_sharpe_ratio(
    SR=sr, 
    SR0=0, 
    T=24, 
    gamma3=-2.448,  # Negative skewness
    gamma4=10.164   # Excess kurtosis
)
print(f"PSR (adjusted): {psr:.3f}")  # Output: PSR (adjusted): 0.987
```

### Minimum Track Record Length

How long must you observe a strategy to claim it's significantly better than a benchmark?

```python
from jsharpe import minimum_track_record_length

# Strategy with SR=0.5, testing against SR0=0 at 95% confidence
months_needed = minimum_track_record_length(SR=0.5, SR0=0, alpha=0.05)
print(f"Months needed: {months_needed:.1f}")
```

### Testing Multiple Strategies

When testing many strategies, control the False Discovery Rate:

```python
from jsharpe import control_for_FDR

# Test 10 strategies, controlling FDR at 25%
alpha, beta, SR_critical, q_hat = control_for_FDR(
    q=0.25,           # Target FDR
    SR0=0,            # Null hypothesis
    SR1=0.5,          # Alternative hypothesis
    p_H1=0.05,        # Prior prob of true signal
    T=24              # Observations per strategy
)

print(f"Critical SR threshold: {SR_critical:.3f}")
print(f"Only accept strategies with SR > {SR_critical:.3f}")
```

### Variance of Sharpe Ratio Estimates

```python
from jsharpe import sharpe_ratio_variance
import math

# Variance under Gaussian assumptions
var_gaussian = sharpe_ratio_variance(SR=0.5, T=24)
print(f"Std error (Gaussian): {math.sqrt(var_gaussian):.3f}")

# Variance with fat tails (higher kurtosis)
var_fat_tails = sharpe_ratio_variance(SR=0.5, T=24, gamma4=6.0)
print(f"Std error (fat tails): {math.sqrt(var_fat_tails):.3f}")
```

```result
PSR: 0.987
PSR (adjusted): 0.987
Months needed: 10.8
Critical SR threshold: 0.479
Only accept strategies with SR > 0.479
Std error (Gaussian): 0.217
Std error (fat tails): 0.234
```

## Core Functions

- `probabilistic_sharpe_ratio()` - Compute PSR with various adjustments
- `sharpe_ratio_variance()` - Variance of SR estimator under non-Gaussian returns
- `minimum_track_record_length()` - Min observations for significance
- `critical_sharpe_ratio()` - Threshold for hypothesis testing
- `sharpe_ratio_power()` - Statistical power of SR test
- `control_for_FDR()` - False Discovery Rate control for multiple testing
- `adjusted_p_values_bonferroni()` - Bonferroni correction
- `adjusted_p_values_holm()` - Holm's step-down procedure
- `adjusted_p_values_sidak()` - Šidák correction
- `minimum_variance_weights_for_correlated_assets()` - Portfolio optimization

## Documentation

- **[API Documentation](https://tschm.github.io/jsharpe)** - Complete API reference with detailed function documentation
- **[Interactive Notebooks](book/marimo/)** - Explore PSR concepts with interactive Marimo notebooks

## References

This library implements methods from:

- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.

## For Developers

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/tschm/jsharpe.git
cd jsharpe

# Install dependencies and setup environment
make install
```

This installs [uv](https://github.com/astral-sh/uv), creates a virtual environment, and installs all dependencies.

### Development Workflow

```bash
# Run tests
make test

# Format code
make fmt

# Start interactive notebooks
make marimo
```

### Project Structure

```
jsharpe/
├── src/jsharpe/           # Main package source code
│   ├── __init__.py        # Top-level public API facade
│   └── sharpe/            # Topical sub-modules (see ARCHITECTURE.md)
│       ├── __init__.py    # Subpackage public API facade
│       ├── linalg.py      # ppoints + covariance helpers (base layer)
│       ├── quadrature.py  # Gauss–Hermite expectation + moments (base layer)
│       ├── clustering.py  # effective rank + clustering (self-contained)
│       ├── psr.py         # Sharpe variance, track record, PSR, power
│       ├── corrections.py # FWER / FDR multiple-testing corrections
│       └── generators.py  # synthetic data + autocorrelation
├── tests/jsharpe/sharpe/  # Unit tests mirroring the source layout 1:1
├── tests/fuzz/            # Atheris fuzz harness
├── book/marimo/           # Marimo notebooks for exploration
├── ARCHITECTURE.md        # Module layering and facade contract
└── pyproject.toml         # Project metadata and dependencies
```

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run `make test` and `make fmt`
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_sharpe.py -v
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use jsharpe in your research, please cite:

```bibtex
@software{jsharpe,
  author = {Thomas Schmelzer},
  title = {jsharpe: Probabilistic Sharpe Ratio and Statistical Testing},
  year = {2024},
  url = {https://github.com/tschm/jsharpe}
}
```
