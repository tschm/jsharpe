# 📦 [jsharpe]()

[![PyPI version](https://badge.fury.io/py/jsharpe.svg)](https://badge.fury.io/py/jsharpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tschm/jsharpe/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/jsharpe/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/tschm/jsharpe/badge.svg?branch=main)](https://coveralls.io/github/tschm/jsharpe?branch=main)
[![Created with qCradle](https://img.shields.io/badge/Created%20with-qCradle-blue?style=flat-square)](https://github.com/tschm/package)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/jsharpe)

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

### **📝 Update Project Info**

- Edit `pyproject.toml` to update authors and email addresses
- Configure GitHub Pages (branch: gh-pages) in repository settings

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

## ⚠️ Trusted publishing failure

That's good news!

You are not able to publish to PyPI unless you have registered your project
on PyPI. You get the following message:

```bash
Trusted publishing exchange failure:

Token request failed: the server refused the request for
the following reasons:

invalid-publisher: valid token, but no corresponding
publisher (All lookup strategies exhausted)
This generally indicates a trusted publisher
configuration error, but could
also indicate an internal error on GitHub or PyPI's part.

The claims rendered below are for debugging purposes only.
You should not
use them to configure a trusted publisher unless they
already match your expectations.
```

Please register your repository. The 'release.yml' flow is
publishing from the 'release' environment. Once you have
registered your new repo it should all work.
