## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: install the package itself (non-editable) so
# mkdocstrings can import `jsharpe` to render the API reference, plus the
# mkdocstrings[python] plugin. The package uses a src/ layout, so without
# `--with .` the ephemeral uvx build env lacks it and `make book` fails with
# ModuleNotFoundError. We use `--with .` (not `--with-editable .`) to honour
# the template's no-editable-install policy.
MKDOCS_EXTRA_PACKAGES = --with . --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
