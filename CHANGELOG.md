# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.6.2] - 2026-06-18

### Bug Fixes
- Bump rhiza_benchmark.yml reference to v0.18.4 (#234)
- Install jsharpe (non-editable) so mkdocstrings can import it in book build (#244)

### Maintenance
- Update rhiza to v0.15.1 (#226)
- Update rhiza to v0.15.2 (#227)
- Resolve rhiza v0.17.0 sync conflicts
- Add pip dependabot entry for .rhiza/requirements
- Update rhiza to v0.18.4 (#231)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#230)
- Apply rhiza sync v0.18.4 (#233)
- Chore(deps)(deps): bump the github-actions group across 1 directory with 8 updates (#232)
- Chore(deps-dev)(deps-dev): bump plotly in the python-dependencies group (#236)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#235)
- Apply rhiza sync v0.18.8 (#237)
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#240)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group across 1 directory with 2 updates (#241)
- Cover number_of_clusters edge cases for 100% coverage (#243)

### Other Changes
- Update template.yml to remove unused templates
- Sync Rhiza template v0.18.8 → v0.19.3 (#242)

## [0.6.1] - 2026-05-21

### Dependencies
- *(deps)* Update dependency astral-sh/uv to v0.11.8 (#214)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group across 1 directory with 4 updates (#223)

### Other Changes
- Add CONTRIBUTING.md to fix broken link in README (#213)
- Update dependency marimo to v0.23.4 (#216)
- Update github/codeql-action action to v4.35.5 (#217)
- Update dependency marimo to v0.23.6 (#218)
- Update dependency astral-sh/uv to v0.11.15 (#221)
- Bump version 0.6.0 → 0.6.1

## [0.6.0] - 2026-04-25

### Dependencies
- *(deps)* Update dependency marimo to v0.23.3 (#210)

### Other Changes
- Add number_of_clusters without introducing new dependencies (#208)
- Delete docs/development directory (#209)
- Bump version 0.5.0 → 0.6.0

## [0.5.0] - 2026-04-22

### Dependencies
- *(deps)* Update dependency astral-sh/uv to v0.11.2 (#164)
- *(deps)* Update actions/deploy-pages action to v5 (#178)
- *(deps)* Update github/codeql-action action to v4.35.1 (#160)
- *(deps)* Update astral-sh/setup-uv action to v8 (#179)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#177)
- Chore(deps)(deps): bump numpy in the python-dependencies group (#181)
- Sync with rhiza template v0.9.5 (#192)
- Update rhiza template ref to v0.10.1 (#197)
- Chore(deps)(deps): bump astral-sh/setup-uv in the github-actions group (#199)
- Sync with rhiza template v0.10.1 (#198)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group across 1 directory with 3 updates (#193)
- Sync with rhiza template v0.10.2 (#200)

### Other Changes
- Update mkdocs.yml
- Update README.md
- Rhiza/update template v0.10.2 (#201)
- Fix coverage badge URL to use GitHub Pages endpoint (#203)
- Test (#204)
- Update MARIMO_FOLDER path in .env file
- Test (#205)
- Bump version 0.4.6 → 0.5.0

## [0.4.6] - 2026-03-22

### Documentation
- Add downloaded reference papers
- Add docstrings to inner functions for 100% docs coverage

### Dependencies
- *(deps)* Update dependency marimo to v0.21.1 (#173)

### Maintenance
- Increase coverage to 100%

### Other Changes
- Update template.yml
- Sync
- Update license to MIT and specify license files
- [WIP] Fix broken badge to point to new URL (#175)
- Bump version 0.4.5 → 0.4.6

## [0.4.5] - 2026-03-17

### Dependencies
- *(deps)* Update dependency astral-sh/uv to v0.10.7 (#156)
- *(deps)* Update astral-sh/setup-uv action to v7.3.1 (#158)
- *(deps)* Update dependency marimo to v0.21.0 (#163)

### Maintenance
- Update via rhiza (#170)

### Other Changes
- Fix coverage badge by generating shields.io endpoint JSON in test target (#155)
- Update template reference to version 0.8.5 (#157)
- Remove history
- Update template.yml reference and templates (#172)
- Bump version 0.4.4 → 0.4.5

## [0.4.4] - 2026-02-27

### Dependencies
- *(deps)* Update actions/upload-artifact action to v4.6.2 (#148)
- *(deps)* Update dependency astral-sh/uv to v0.10.6 (#149)
- *(deps)* Update github artifact actions (#153)
- *(deps)* Update actions/attest-sbom action to v4 (#152)
- *(deps)* Update actions/attest-build-provenance action to v4 (#151)

### Other Changes
- Sync
- Add renovate.json (#147)
- Bump version 0.4.3 → 0.4.4

## [0.4.3] - 2026-02-27

### Bug Fixes
- Replace uvx hatch build with uv build to avoid virtualenv incompatibility

### Other Changes
- Bump version 0.4.2 → 0.4.3

## [0.4.2] - 2026-02-27

### Dependencies
- *(deps)* Lock file maintenance (#122)
- *(deps)* Update dependency marimo to v0.19.11 (#124)
- *(deps)* Lock file maintenance (#125)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.11 (#126)
- *(deps)* Update actions/download-artifact action to v7 (#130)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.0 (#129)
- *(deps)* Update dependency astral-sh/uv to v0.10.3 (#131)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.3 (#132)
- *(deps)* Update dependency astral-sh/uv to v0.10.4 (#133)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.4 (#134)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2 (#135)
- *(deps)* Update github/codeql-action action to v4.32.4 (#136)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.2 (#137)
- *(deps)* Update dependency marimo to v0.20.1 (#138)
- *(deps)* Update dependency marimo to v0.20.2 (#139)
- *(deps)* Lock file maintenance (#140)
- *(deps)* Update dependency astral-sh/uv to v0.10.6 (#143)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.6 (#144)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.4 (#145)
- *(deps)* Update pre-commit hook pycqa/bandit to v1.9.4 (#146)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.3 (#142)

### Other Changes
- Sync
- Fmt
- Update template repository and branch reference
- Sync
- Fmt
- Bump version 0.4.1 → 0.4.2

## [0.4.1] - 2026-02-07

### Dependencies
- *(deps)* Lock file maintenance (#105)
- *(deps)* Update github/codeql-action action to v4.32.1 (#108)
- *(deps)* Update dependency marimo to v0.19.7 (#107)
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.25 (#109)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.30 (#111)
- *(deps)* Update astral-sh/setup-uv action to v7.3.0 (#113)
- *(deps)* Update github/codeql-action action to v4.32.2 (#112)
- *(deps)* Update dependency astral-sh/uv to v0.10.0 (#114)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.0 (#115)
- *(deps)* Lock file maintenance (#116)
- *(deps)* Update dependency marimo to v0.19.9 (#110)

### Maintenance
- Update via rhiza (#106)

### Other Changes
- Add comprehensive comparison analysis with zoonek/2025-sharpe-ratio
- Add Rhiza infrastructure analysis to repository comparison
- Enhance Rhiza analysis with economics and ROI calculations
- Update repository-comparison.md
- Delete presentation directory (#118)
- Modify template.yml to update included templates (#119)
- Psr notebook
- Bump version 0.4.0 → 0.4.1

## [0.4.0] - 2026-01-30

### Dependencies
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.0 (#71)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.10 (#74)
- *(deps)* Lock file maintenance (#73)
- *(deps)* Lock file maintenance (#77)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.20 (#79)
- *(deps)* Update dependency astral-sh/uv to v0.9.20 (#80)
- *(deps)* Lock file maintenance (#85)
- *(deps)* Update dependency astral-sh/uv to v0.9.22 (#87)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.22 (#88)
- *(deps)* Lock file maintenance (#89)
- *(deps)* Lock file maintenance (#90)
- *(deps)* Update dependency marimo to v0.19.2 (#92)
- *(deps)* Lock file maintenance (#95)
- *(deps)* Update pre-commit hook pycqa/bandit to v1.9.3 (#98)
- *(deps)* Lock file maintenance (#99)
- *(deps)* Update dependency marimo to v0.19.6 (#101)
- *(deps)* Update dependency astral-sh/uv to v0.9.27 (#103)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.27 (#104)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.1 (#102)

### Maintenance
- Sync template from jebel-quant/rhiza@main (#70)
- Test_cli_commands
- Remove deprecated files (#72)
- Update via rhiza (#78)
- Update via rhiza (#86)
- Update via rhiza (#91)
- Chore(deps-dev)(deps-dev): bump marimo (#97)
- Sync with rhiza (#96)
- Update via rhiza (#100)

### Other Changes
- Rhiza as dev dependency
- Fix formatting of include and exclude lists in template
- Fmt
- Rhiza
- Clean up pyproject.toml by removing comments
- Template for book and presentation
- Rhiza
- Rhiza folder in .github
- Rhiza folder in .github
- Will come back later
- Rhiza
- Rhiza
- Migrate
- Materialize
- Migrate
- Suspicious files
- Rhiza
- Delete .github/workflows/structure.yml
- Fix incorrect paths and add comprehensive directory structure to README (#76)
- Sync
- Deptry
- Lock file"
- Enhance README header with comprehensive badges and project description (#82)
- Update README.md
- Update README.md
- Update README.md
- Rewrite README.md with user-focused content and concise developer section (#84)
- Update README.md
- Sync
- Sync
- Rhiza sync
- Delete .rhiza.env (#93)
- Sync
- Rename book/marimo/psr.py to book/marimo/notebooks/psr.py
- Update psr.py
- Sync
- Bump version 0.3.0 → 0.4.0

## [0.3.0] - 2025-12-15

### Bug Fixes
- *(deps)* Update dependency marimo to v0.18.3
- *(deps)* Update dependency marimo to v0.18.4
- Fixing Makefile
- Fix notebook

### Documentation
- Add section about config-templates in README
- Clarify that some tests are from config-templates

### Dependencies
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#42)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.17 (#36)
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.47.0 (#62)
- *(deps)* Lock file maintenance (#68)

### Maintenance
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files (#63)
- Sync template from jebel-quant/rhiza@main (#64)
- Sync template files (#66)
- Sync template files (#67)
- Sync template files (#69)

### Other Changes
- Sync
- Merge pull request #39 from tschm/renovate/marimo-0.x
- Merge pull request #40 from tschm/renovate/lock-file-maintenance
- Update template.yml
- Devcontainer
- Logging in test_sharpe
- Logging in test_sharpe
- Remove dated tests
- Merge pull request #41 from tschm/template-updates
- Merge pull request #43 from tschm/template-updates
- Update template.yml
- Delete .github/workflows/devcontainer.yml
- Delete .devcontainer directory
- Merge pull request #45 from tschm/template-updates
- Initial plan
- Merge pull request #47 from tschm/copilot/update-readme-with-boilerplate-info
- Revise README with usage examples and configuration
- Refactor README to eliminate duplicate content
- Revise development commands and contributing guidelines
- Fix typo in Development Tools list in README
- Update links in README for documentation and config
- Update README.md
- Merge pull request #48 from tschm/tschm-patch-1
- Remove obsolete conftest
- Merge pull request #50 from tschm/renovate/marimo-0.x
- Merge pull request #51 from tschm/renovate/lock-file-maintenance
- Initial plan
- Add sys.path configuration to marimo notebook to find jsharpe package
- Improve path resolution to use pyproject.toml as marker file
- Refactor path resolution to use more Pythonic approach with next()
- Add the src path to sys path
- Merge pull request #52 from tschm/copilot/fix-notebook-path-issue
- Install jsharpe in notebook
- Fmt
- Install jsharpe in notebook
- Install jsharpe in notebook
- Install jsharpe in notebook
- Fmt
- Merge pull request #53 from tschm/install
- Install jsharpe in notebook
- Install jsharpe in notebook
- Install jsharpe in notebook
- Install jsharpe in notebook
- Fmt
- Merge pull request #54 from tschm/install
- Wait for subprocess to finish
- Wait for subprocess to finish
- Fmt
- Merge pull request #55 from tschm/install
- Wait for subprocess to finish
- Wait for subprocess to finish
- Fmt
- Fmt
- Fmt
- Merge pull request #56 from tschm/install
- Merge pull request #57 from tschm/template-updates
- Wait for subprocess to finish
- Tool.uv.source
- Tool.uv.source
- Tool.uv.source
- Fmt
- Tool.uv.source
- Simplify pyproject
- Update .github/scripts/marimushka.sh
- Merge pull request #58 from tschm/mari
- Tool.uv.source
- Install jsharpe in notebook
- Install jsharpe in notebook
- Install jsharpe in notebook
- Install jsharpe in notebook
- Merge pull request #59 from tschm/chatgpt
- Marimushka
- Install jsharpe in notebook
- Install jsharpe in notebook
- Bring in the template
- Fmt
- Fmt
- Force install package
- Fmt
- Merge pull request #60 from tschm/chatgpt
- Update webpage generation (#61)
- README update for templates
- Delete tests/test_config_templates directory
- Change template repository to jebel-quant/rhiza
- Delete .github/scripts/post-release.sh (#65)
- Remove traces of tschm/.config-templates
- Delete .github/templates directory
- Delete .github/scripts/build-extras.sh

## [0.2.0] - 2025-12-07

### Maintenance
- Test bumping

### Other Changes
- Remove bump test

## [0.1.0] - 2025-12-05

### Bug Fixes
- *(deps)* Update dependency marimo to v0.18.2

### Maintenance
- Sync template files

### Other Changes
- Merge pull request #32 from tschm/template-updates
- Merge pull request #33 from tschm/renovate/marimo-0.x
- Remove the test_release_script
- Notebook!
- Fmt
- Workflows
- Missing scripts
- Sync

## [0.0.2] - 2025-12-03

### Other Changes
- Update README.md

## [0.0.1] - 2025-12-03

### Bug Fixes
- Fix fmt in __init__
- *(deps)* Update dependency marimo to v0.18.0 (#11)
- *(deps)* Update dependency marimo to v0.18.1 (#21)
- Fixing demo
- Fixing "make docs"
- Fix notebook

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.6 (#4)
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.46.0 (#8)
- *(deps)* Update actions/checkout action to v6 (#9)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.9 (#5)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.11 (#3)
- *(deps)* Lock file maintenance (#12)
- *(deps)* Lock file maintenance (#13)
- *(deps)* Lock file maintenance (#17)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.14 (#20)
- *(deps)* Update softprops/action-gh-release action to v2.5.0

### Maintenance
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Test minimum variance
- Tests into one file
- Tests
- Sync template files

### Other Changes
- First commit
- Dependencies
- Bring the ppoint fct
- Reduce dependencies
- Pyproject
- Merge pull request #14 from tschm/template-updates
- Update README.md
- Merge pull request #16 from tschm/template-updates
- Remove tests/test_taskfile.py Taskfile.yml taskfiles
- Merge pull request #15 from tschm/remove-file-6
- Merge pull request #18 from tschm/template-updates
- Update template.yml to modify included and excluded files
- Merge pull request #19 from tschm/tschm-patch-1
- Merge pull request #22 from tschm/renovate/softprops-action-gh-release-2.x
- Remove docker
- Remove devcontainer
- Delete .devcontainer directory
- Merge pull request #23 from tschm/tschm-patch-1
- Merge pull request #24 from tschm/template-updates
- Ignore the original functions.py file
- Moving fraction of functions over
- Initial plan
- Fix all ruff linting issues
- Address code review feedback
- Merge pull request #26 from tschm/copilot/address-ruff-issues
- Remove scikit-learn and matplotlib
- Fmt
- Remove pandas
- Fmt
- README
- Sharpe
- Initial plan
- Revisit docstrings, add code examples, follow google-style docs
- Do not update ruff.toml
- Directly import from jsharpe
- Merge branch 'main' into copilot/revisit-docstrings
- Merge pull request #28 from tschm/copilot/revisit-docstrings
- Update documentation generation command in Makefile
- Fmt
- Merge pull request #30 from tschm/template-updates
- Update test_release_script.py
- Simplify example by removing sys.path.append
- README without sys construction
- Clean up README by removing unnecessary sections

<!-- generated by git-cliff -->
