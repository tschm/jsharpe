# Configure Git Auth for Private Packages

This composite action configures git to use token authentication for private GitHub packages.

## Usage

Add this step before installing dependencies that include private GitHub packages:

```yaml
- name: Configure git auth for private packages
  uses: ./.github/actions/configure-git-auth
  with:
    token: ${{ secrets.GH_PAT }}
```

The `GH_PAT` secret should be a Personal Access Token with `repo` scope.

## What It Does

This action runs:

```bash
git config --global url."https://<token>@github.com/".insteadOf "https://github.com/"
```

This tells git to automatically inject the token into all HTTPS GitHub URLs, enabling access to private repositories.

## When to Use

Use this action when your project has dependencies defined in `pyproject.toml` like:

```toml
[tool.uv.sources]
private-package = { git = "https://github.com/your-org/private-package.git", rev = "v1.0.0" }
```

## Token Requirements

The default `GITHUB_TOKEN` does **not** have access to other private repositories, even within the same organization. You must use a Personal Access Token (PAT) with `repo` scope stored as `secrets.GH_PAT`.

If `secrets.GH_PAT` is not defined, GitHub Actions passes an empty string. The action will still run successfully, but any subsequent step that tries to access private repositories will fail with an authentication error.

## Example Workflow

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Configure git auth for private packages
        uses: ./.github/actions/configure-git-auth
        with:
          token: ${{ secrets.GH_PAT }}

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run tests
        run: uv run pytest
```

## See Also

- [PRIVATE_PACKAGES.md](../../../.rhiza/docs/PRIVATE_PACKAGES.md) - Complete guide to using private packages
- [TOKEN_SETUP.md](../../../.rhiza/docs/TOKEN_SETUP.md) - Setting up Personal Access Tokens
