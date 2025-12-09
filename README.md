# 📦 [jsharpe](https://github.com/tschm/jsharpe)

[![PyPI version](https://badge.fury.io/py/jsharpe.svg)](https://badge.fury.io/py/jsharpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tschm/jsharpe/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/jsharpe/actions/workflows/ci.yml)
[![Created with qCradle](https://img.shields.io/badge/Created%20with-qCradle-blue?style=flat-square)](https://github.com/tschm/package)

## 🚀 Getting Started

### **🔧 Set Up Environment**

```bash
make install
```

This installs/updates [uv](https://github.com/astral-sh/uv),
creates your virtual environment and installs dependencies.

For adding or removing packages:

```bash
uv add/remove requests  # for main dependencies
uv add/remove requests --dev  # for dev dependencies
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

This project uses standardized configuration files from [tschm/.config-templates](https://github.com/tschm/.config-templates), which provides a consistent development environment across multiple projects.

### Synchronized Files

The following files are automatically synchronized from the template repository:

- **Development Tools**: `.editorconfig`, `.pre-commit-config.yaml`, `Makefile`, `ruff.toml`, `pytest.ini`
- **GitHub Workflows**: CI/CD pipelines in `.github/workflows/`
- **Documentation**: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `LICENSE`
- **Configuration**: `.gitignore` and other project setup files

### Template Synchronization

The `.github/template.yml` file controls which files are synchronized from the template repository. To sync with the latest template updates:

```bash
make sync
```

This ensures the project benefits from improvements to the shared configuration without manual updates.

### Customization

While most boilerplate files come from the template, the following are project-specific:
- `README.md` (this file)
- `pyproject.toml` (project dependencies and metadata)
- `ruff.toml` (extended but based on template)
- Source code in `src/` and tests in `tests/`

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
