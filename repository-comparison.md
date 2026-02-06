# Repository Comparison: jsharpe vs 2025-sharpe-ratio

## Executive Summary

This analysis compares **tschm/jsharpe** and **zoonek/2025-sharpe-ratio** - two repositories focused on Sharpe ratio analysis and statistical testing. While both implement similar statistical concepts from Marcos Lopez de Prado's research, they differ significantly in their purpose, implementation quality, and maintainability.

**Key Findings:**
- **jsharpe**: A production-ready Python package with superior code quality, comprehensive testing, and professional infrastructure
- **2025-sharpe-ratio**: A research companion repository with useful notebooks but limited code organization and maintainability

---

## 1. Repository Overview

### tschm/jsharpe
- **Purpose**: Production Python library for Sharpe ratio analysis
- **Primary Language**: Python (package)
- **Created**: November 2025
- **Stars**: 19 | **Forks**: 4
- **License**: MIT
- **Homepage**: https://tschm.github.io/jsharpe/
- **Status**: Actively maintained, version 0.4.0

### zoonek/2025-sharpe-ratio
- **Purpose**: Code companion for academic paper "Sharpe Ratio Inference: A New Standard for Reporting and Decision-Making"
- **Primary Language**: Jupyter Notebook
- **Created**: September 2025
- **Stars**: 86 | **Forks**: 16
- **License**: None specified
- **Homepage**: None
- **Status**: Research/educational code

---

## 2. Code Quality & Practices

### Architecture & Organization

**jsharpe** ⭐⭐⭐⭐⭐
```
✓ Clean modular structure (src/jsharpe/)
✓ Separation of concerns (single sharpe.py module)
✓ Clear public API via __init__.py
✓ Proper Python package layout with pyproject.toml
✓ Type hints throughout
✓ Comprehensive docstrings (Google style)
```

**2025-sharpe-ratio** ⭐⭐
```
✗ Single monolithic functions.py file (1,288 lines)
✗ Mixed concerns (utility functions, tests, data generation, plotting)
✗ No package structure
✗ Jupyter notebooks for all examples (8 notebooks)
✗ Limited modularity
```

### Code Style & Documentation

**jsharpe** ⭐⭐⭐⭐⭐
```python
# Example from jsharpe/sharpe.py
def probabilistic_sharpe_ratio(
    SR: float,
    SR0: float,
    *,
    variance: float | None = None,
    T: int | None = None,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR).

    The PSR is 1 - p, where p is the p-value of testing H0: SR = SR0 vs
    H1: SR > SR0. It can be interpreted as a "Sharpe ratio on a probability
    scale", i.e., mapping the SR to [0, 1].

    Args:
        SR: Observed Sharpe ratio.
        SR0: Sharpe ratio under null hypothesis.
        variance: Variance of SR estimator. Provide this OR (T, gamma3, ...).
        T: Number of observations (if variance not provided).
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        K: Number of strategies for variance adjustment. Default 1.

    Returns:
        Probabilistic Sharpe Ratio in [0, 1]. Values near 1 indicate
        strong evidence that the true SR exceeds SR0.

    Example:
        >>> psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0, T=24)
        >>> bool(0 < psr < 1)
        True
    """
```

**2025-sharpe-ratio** ⭐⭐⭐
```python
# Example from functions.py
def probabilistic_sharpe_ratio(
    SR: float,
    SR0: float,
    *,
    variance: float = None,
    T: int = None,
    gamma3: float = 0.,
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR)

    This is 1-p, where p is the p-value of the test  H0: SR=SR0  vs  H1: SR>SR0.
    It can be interpreted as a Sharpe ratio "on a probability scale", i.e., in [0,1].

    TODO: In case of multiple testing, we currently expect SR0 to be already adjusted; this function will only adjust the variance

    Inputs:
    - SR: float, observed Sharpe ratio
    - SR0: float, Sharpe ratio under H0
    - T: int, number of observations
    # [truncated for brevity]
    """
```

**Comparison:**
- ✓ jsharpe: Modern type hints (e.g., `float | None`), comprehensive docstrings with examples
- ✗ 2025-sharpe-ratio: Older style (no union types), less structured documentation, contains TODOs in production code

### Code Metrics

| Metric | jsharpe | 2025-sharpe-ratio |
|--------|---------|-------------------|
| **Core Source Lines** | ~1,000 LOC | ~1,288 LOC |
| **Test Lines** | ~544 LOC | Inline tests only |
| **Test Coverage** | Comprehensive (badge shows coverage) | Minimal |
| **Files Structure** | 2 source files, clean separation | 1 monolithic file + 8 notebooks |
| **Functions Exported** | 24 public functions | ~60+ functions (many internal) |
| **Deprecated Code** | None | Several `@deprecated` functions still present |

---

## 3. Testing & Quality Assurance

### Test Infrastructure

**jsharpe** ⭐⭐⭐⭐⭐
```
✓ Comprehensive pytest test suite (544 LOC)
✓ 40+ test functions covering edge cases
✓ Parametric tests for multiple scenarios
✓ Coverage tracking and reporting
✓ CI runs tests on multiple Python versions (3.11-3.14)
✓ Fixtures and proper test organization
✓ Tests for numerical accuracy and edge cases
```

Example test structure:
```python
def test_FDR_critical_value_edge_returns():
    """Explicitly trigger -inf and NaN branches in FDR_critical_value."""
    # For c -> -inf, f(-10) ≈ 1 - p. If q > 1 - p then function returns -inf
    mu0, mu1 = 0.0, 1.0
    sigma0, sigma1 = 1.0, 1.0
    p = 0.2
    q = 0.85  # > 1 - p = 0.8
    c = FDR_critical_value(q, mu0, mu1, sigma0, sigma1, p)
    assert c == -np.inf
    # ... more edge cases
```

**2025-sharpe-ratio** ⭐⭐
```
✗ No separate test suite
✗ Tests embedded in main file (if __name__ == '__main__')
✗ No CI/CD
✗ No coverage tracking
✗ Basic assertion tests only
✗ No edge case testing
```

Example test structure:
```python
if __name__ == '__main__':
    test_effective_rank()
    test_sharpe_ratio_variance()
    test_minimum_track_record_length()
    # ... simple function calls
    print("All tests passed.")
```

### Quality Checks

**jsharpe** ⭐⭐⭐⭐⭐
```
✓ 14 GitHub Actions workflows
✓ Pre-commit hooks
✓ Type checking (mypy)
✓ Linting and formatting
✓ Security scanning (CodeQL, dependency scanning)
✓ Documentation coverage checks
✓ Benchmark testing
✓ CodeFactor: A rating
```

**2025-sharpe-ratio** ⭐
```
✗ No CI/CD
✗ No automated quality checks
✗ Basic .flake8 config present but not automated
✗ No security scanning
✗ No type checking
```

---

## 4. Documentation & Usability

### Documentation Quality

**jsharpe** ⭐⭐⭐⭐⭐
```
✓ Professional README with badges and examples
✓ API documentation site (https://tschm.github.io/jsharpe/)
✓ Interactive Marimo notebooks for exploration
✓ Code examples in docstrings
✓ Contributing guidelines (CONTRIBUTING.md)
✓ Code of Conduct (CODE_OF_CONDUCT.md)
✓ Citation information
✓ Clear installation instructions
```

**2025-sharpe-ratio** ⭐⭐⭐
```
✓ Minimal but functional README
✓ Paper reference and context
✗ No API documentation
✓ Jupyter notebooks demonstrate usage
✗ No contributing guidelines
✗ No citation format
✗ No code of conduct
```

### User Experience

**jsharpe**:
```bash
# Installation
pip install jsharpe

# Usage
from jsharpe import probabilistic_sharpe_ratio
psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0, T=24)
```
→ **Clean, professional package experience**

**2025-sharpe-ratio**:
```bash
# Installation (manual)
uv venv
uv pip install scikit-learn scipy statsmodels matplotlib seaborn tqdm cvxpy ray deprecated papermill ipykernel ipywidgets
uv run functions.py  # Tests, and the numeric example from the paper
```
→ **Research code experience, requires manual setup**

---

## 5. Dependency Management

### jsharpe ⭐⭐⭐⭐⭐
```toml
# pyproject.toml
dependencies = [
    "numpy>=2.3.0",
    "scipy>=1.16.3",
]

[dependency-groups]
dev = [
    "marimo==0.19.6",
    "cvxpy>=1.7.0",
    "jinja2>=3.1.6",
    "plotly>=6.5",
    "pandas>=2.3"
]
```
**Strengths:**
- Minimal runtime dependencies (only numpy, scipy)
- Modern pyproject.toml configuration
- Separation of runtime vs dev dependencies
- Version pinning for reproducibility
- Compatible with modern Python (3.11-3.14)

### 2025-sharpe-ratio ⭐⭐
```txt
# requirements.txt (95 lines - all dependencies)
anyio==4.8.0
argon2-cffi==23.1.0
# ... 93 more lines including Jupyter stack
```
**Weaknesses:**
- No separation of runtime vs dev dependencies
- All Jupyter infrastructure in single requirements file
- Heavy dependency footprint
- No pyproject.toml
- Unclear what's actually needed for core functions

---

## 6. Maintainability & Best Practices

### Software Engineering Practices

| Practice | jsharpe | 2025-sharpe-ratio |
|----------|---------|-------------------|
| **Version Control** | ✅ Semantic versioning | ⚠️ No versioning |
| **Change Management** | ✅ Structured releases | ❌ No release process |
| **Issue Tracking** | ✅ Enabled | ✅ Enabled |
| **Pull Requests** | ✅ Required, CI checks | ⚠️ No protection |
| **Code Review** | ✅ Automated + manual | ❌ None |
| **Security** | ✅ Dependabot, CodeQL | ❌ None |
| **Licensing** | ✅ MIT License | ❌ No license |

### Code Health Indicators

**jsharpe:**
- ✅ No deprecated functions
- ✅ Clear error handling
- ✅ Type safety
- ✅ Modular design enables testing
- ✅ Single responsibility functions
- ✅ Consistent naming conventions

**2025-sharpe-ratio:**
- ⚠️ Multiple `@deprecated` functions still in codebase
- ⚠️ Mix of styles (unicode symbols vs ASCII)
- ⚠️ Some functions >100 LOC
- ⚠️ TODO comments in production code
- ⚠️ Hard-coded data in source files
- ⚠️ Global state (E_under_normal)

---

## 7. Feature Comparison

### Core Features

Both repositories implement similar statistical methods:

| Feature | jsharpe | 2025-sharpe-ratio |
|---------|---------|-------------------|
| **Sharpe Ratio Variance** | ✅ | ✅ |
| **Probabilistic SR** | ✅ | ✅ |
| **Minimum Track Record Length** | ✅ | ✅ |
| **Critical SR** | ✅ | ✅ |
| **Power Calculation** | ✅ | ✅ |
| **FDR Control** | ✅ | ✅ |
| **FWER Adjustments** | ✅ (Bonferroni, Šidák, Holm) | ✅ (Bonferroni, Šidák, Holm) |
| **Multiple Testing** | ✅ | ✅ |
| **Portfolio Optimization** | ✅ | ✅ |
| **Clustering Analysis** | ❌ | ✅ (silhouette-based) |
| **Data Generation** | ✅ (limited) | ✅ (extensive) |

### Unique Features

**jsharpe:**
- Published package on PyPI
- API documentation website
- Interactive Marimo notebooks
- Production-ready error handling

**2025-sharpe-ratio:**
- Number of clusters computation (k-means + silhouette)
- Deflated Sharpe Ratio (marked deprecated)
- Extensive data generation utilities
- Paper reproduction notebooks
- More visualization examples

---

## 8. Content & Educational Value

### jsharpe ⭐⭐⭐⭐
**Strengths:**
- Clear examples in README
- Progressive complexity in examples
- Docstring examples with doctests
- Focus on practical usage

**Target Audience:** Practitioners, quants, developers building production systems

### 2025-sharpe-ratio ⭐⭐⭐⭐⭐
**Strengths:**
- Direct paper companion (reproduces all figures)
- Detailed mathematical notation in comments
- Extensive visualization notebooks
- Research-oriented examples
- Shows statistical derivations

**Target Audience:** Researchers, academics, students learning the theory

---

## 9. Maintenance & Activity

### Recent Activity

**jsharpe:**
- Last push: February 3, 2026
- Recent commits show active development
- Regular releases (currently 0.4.0)
- Responsive to issues

**2025-sharpe-ratio:**
- Last push: February 5, 2026
- Less frequent updates
- No formal releases
- Academic project maintenance pattern

---

## 10. Recommendations

### For Production Use: Choose **jsharpe**
**Reasons:**
1. ✅ Clean, well-tested codebase
2. ✅ Minimal dependencies
3. ✅ Strong type safety
4. ✅ Comprehensive CI/CD
5. ✅ Active maintenance
6. ✅ Clear licensing
7. ✅ Professional documentation
8. ✅ Easy to integrate (`pip install jsharpe`)

### For Research/Learning: Consider **2025-sharpe-ratio**
**Reasons:**
1. ✅ Paper companion with full reproducibility
2. ✅ Extensive examples and visualizations
3. ✅ Additional features (clustering, etc.)
4. ✅ Higher GitHub stars (academic visibility)

### For New Projects: Strongly recommend **jsharpe**
**Reasons:**
- More maintainable long-term
- Better suited for integration into larger systems
- Lower technical debt
- Easier to extend and modify
- Better documentation for developers

---

## 11. Quantitative Comparison Matrix

| Category | Weight | jsharpe | 2025-sharpe-ratio | Winner |
|----------|--------|---------|-------------------|--------|
| **Code Quality** | 25% | 95/100 | 65/100 | jsharpe |
| **Testing** | 20% | 95/100 | 40/100 | jsharpe |
| **Documentation** | 15% | 90/100 | 70/100 | jsharpe |
| **Maintainability** | 20% | 95/100 | 50/100 | jsharpe |
| **Features** | 10% | 80/100 | 85/100 | 2025-sharpe-ratio |
| **Content/Education** | 10% | 75/100 | 95/100 | 2025-sharpe-ratio |
| **Overall** | 100% | **90.5/100** | **63.0/100** | **jsharpe** |

---

## 12. Specific Issues & Concerns

### 2025-sharpe-ratio Issues:

1. **No License**: Legal uncertainty for commercial use
2. **Monolithic Design**: Single 1,288-line file is hard to maintain
3. **No CI/CD**: Breaking changes may go unnoticed
4. **Deprecated Code**: Functions marked deprecated still in main codebase
5. **Heavy Dependencies**: 95+ dependencies for simple statistical functions
6. **Global State**: `E_under_normal` computed at module level
7. **Hard-coded Data**: Autocorrelation lookup tables as string literals

### jsharpe Minor Issues:

1. **Limited Visualization**: Less plotting utilities (design choice)
2. **Fewer Examples**: Could use more notebooks
3. **Smaller Community**: Fewer GitHub stars (but quality > popularity)

---

## 13. Conclusion

**jsharpe** is a professionally developed Python package that demonstrates excellent software engineering practices. It's production-ready, well-tested, and maintainable. The codebase follows modern Python conventions, has comprehensive CI/CD, and provides a clean API for users.

**2025-sharpe-ratio** is a valuable academic companion to a research paper. It successfully reproduces the paper's results and provides educational value through its notebooks. However, its code organization and lack of testing infrastructure make it less suitable for production use.

### Summary Verdict:

- **Quality & Maintainability**: jsharpe wins decisively
- **Academic/Research Value**: 2025-sharpe-ratio has an edge
- **Production Readiness**: jsharpe is the only viable choice
- **Learning Resource**: Both have value, different strengths

### Recommendation:
For any serious software project requiring Sharpe ratio analysis, **jsharpe** is the clear choice. For reproducing academic results or learning the theory, **2025-sharpe-ratio** provides excellent supplementary material.

The ideal scenario might be to **use jsharpe as the dependency** while **referencing 2025-sharpe-ratio** for theoretical understanding and visualization ideas.

---

**Analysis Date**: February 6, 2026
**Analyst**: Claude Code
**Repository Versions**: jsharpe v0.4.0 | 2025-sharpe-ratio (no version)
