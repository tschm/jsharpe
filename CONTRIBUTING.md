# Contributing to jsharpe

Thank you for your interest in contributing to jsharpe! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (installed automatically via `make install`)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/tschm/jsharpe.git
cd jsharpe

# Install dependencies and setup environment
make install
```

## Development Workflow

```bash
# Run tests with coverage
make test

# Format code
make fmt

# Start interactive notebooks
make marimo
```

## Making Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run `make test` and `make fmt`
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_sharpe.py -v
```

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting. Run `make fmt` to automatically format your code before committing.

## Reporting Issues

Please use [GitHub Issues](https://github.com/tschm/jsharpe/issues) to report bugs or request features. When reporting a bug, include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behaviour
- Python version and platform

## License

By contributing to jsharpe, you agree that your contributions will be licensed under the [MIT License](LICENSE).
