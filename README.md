<div align="center">
# <img src="https://raw.githubusercontent.com/Jebel-Quant/rhiza/main/assets/rhiza-logo.svg" alt="Rhiza Logo" width="30" style="vertical-align: middle;"> [jsharpe](https://tschm.github.io/jsharpe)

![Synced with Rhiza](https://img.shields.io/badge/synced%20with-rhiza-2FA4A9?color=2FA4A9)

[![PyPI version](https://badge.fury.io/py/jsharpe.svg)](https://badge.fury.io/py/jsharpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/endpoint?url=https://tschm.github.io/jsharpe/tests/coverage-badge.json)](https://tschm.github.io/jsharpe/tests/html-coverage/index.html)
[![Downloads](https://static.pepy.tech/personalized-badge/jsharpe?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/jsharpe)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/jsharpe/badge)](https://www.codefactor.io/repository/github/tschm/jsharpe)

A Python library for calculating the Probabilistic Sharpe Ratio (PSR) and related statistics, as introduced by Marcos Lopez de Prado. The PSR provides a more robust way to evaluate trading strategies by accounting for the uncertainty in Sharpe ratio estimates.
</div>

## 🚀 Getting Started

### **🔧 Set Up Environment**

```bash
make install
```

This installs/updates [uv](https://github.com/astral-sh/uv),
creates your virtual environment and installs dependencies.

## 📚 Usage

Run this minimal, deterministic example to compute the
Probabilistic Sharpe Ratio (PSR) from the package functions.

```python
from jsharpe import probabilistic_sharpe_ratio

sr = 0.036 / 0.079
psr = probabilistic_sharpe_ratio(SR=sr, SR0=0, T=24, gamma3=-2.448, gamma4=10.164)
print(f"{psr:.3f}")
```

```result
0.987
```

### **✅ Configure Pre-commit Hooks**

```bash
make fmt
```

Installs hooks to maintain code quality and formatting.

## 🛠️ Development Commands

```bash
make tests   # Run test suite
make marimo  # Start Marimo notebooks
```

## 👥 Contributing

- 🍴 Fork the repository
- 🌿 Create your feature branch (git checkout -b feature/amazing-feature)
- 💾 Commit your changes (git commit -m 'Add some amazing feature')
- 🚢 Push to the branch (git push origin feature/amazing-feature)
- 🔍 Open a Pull Request

## 🏗️ Project Structure & Configuration Templates

This project uses standardized configuration files from [jebel-quant/rhiza](https://github.com/jebel-quant/rhiza), which provides a consistent development environment across multiple projects.

### Directory Structure

```
jsharpe/
├── .github/           # GitHub Actions workflows and configurations
├── .rhiza/            # Rhiza template scripts, utilities, and configuration
├── book/              # Documentation and interactive notebooks
│   ├── marimo/        # Marimo interactive notebooks
│   ├── minibook-templates/  # Documentation templates
│   └── pdoc-templates/      # API documentation templates
├── presentation/      # Marp presentation generation system
├── src/jsharpe/       # Main package source code
└── tests/             # Test suite
    └── test_rhiza/    # Template-provided boilerplate tests
```

### Key Directories

- **[.rhiza/](.rhiza/)** - Platform-agnostic scripts and utilities for repository management
  - `scripts/` - Shell scripts for common tasks (book generation, release, etc.)
  - `scripts/customisations/` - Repository-specific customization hooks
  - `utils/` - Python utilities for version management
  - See [.rhiza/CONFIG.md](.rhiza/CONFIG.md) for details

- **[book/](book/)** - Documentation and interactive content
  - Contains Marimo notebooks for interactive demonstrations
  - Documentation templates for minibook and API docs
  - See [book/marimo/README.md](book/marimo/README.md) for notebook details

- **[presentation/](presentation/)** - Marp-based presentation system
  - Converts Markdown to HTML/PDF slides
  - See [presentation/README.md](presentation/README.md) for usage

- **[src/jsharpe/](src/jsharpe/)** - Main package implementation
  - Core functionality for computing Probabilistic Sharpe Ratios

- **[tests/](tests/)** - Test suite
  - `test_sharpe.py` - Project-specific tests
  - `test_rhiza/` - Template-provided tests for validating boilerplate

### Synchronized Files

The following files are automatically synchronized from the template repository:

- **Development Tools**: [.editorconfig](.editorconfig), [.pre-commit-config.yaml](.pre-commit-config.yaml), [Makefile](Makefile), [pytest.ini](pytest.ini)
- **GitHub Workflows**: CI/CD pipelines in [.github/workflows](.github/workflows)
- **Documentation**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [CONTRIBUTING.md](CONTRIBUTING.md), [LICENSE](LICENSE)
- **Configuration**: [.gitignore](.gitignore), [renovate.json](renovate.json), and other project setup files

### Template Synchronization

The [.rhiza/template.yml](.rhiza/template.yml) file controls which files are synchronized from the template repository. To sync with the latest template updates:

```bash
make sync
```

This ensures the project benefits from improvements to the shared configuration without manual updates.

### Customization

While most boilerplate files come from the template, the following are project-specific:
- [README.md](README.md) (this file)
- [pyproject.toml](pyproject.toml) (project dependencies and metadata)
- [ruff.toml](ruff.toml) (extended but based on template)
- Source code in [src/](src/)
- Project-specific tests (e.g., `tests/test_sharpe.py`)

Note: The [tests/test_rhiza](tests/test_rhiza) directory contains template-provided tests for validating the boilerplate configuration itself.


