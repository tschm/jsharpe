# Architecture

`jsharpe` is a small, layered library. This note records the *intended*
module layering for the `jsharpe.sharpe` subpackage so future edits keep the
import graph an acyclic DAG (no cycles, no upward imports).

## Layering

Imports only ever point **downward**. A module in an upper layer may import
lower layers; a lower layer must never import an upper one.

```
        corrections            (FWER / FDR multiple-testing corrections)
             │
             ▼
            psr                (Sharpe variance, track record, PSR, power)
             │
             ▼
         quadrature            (Gauss–Hermite expectation, moments of max)

        generators  ──▶  linalg   (synthetic data ──▶ ppoints / covariance)

        clustering             (effective rank, k-means clustering — self-contained)
```

Concretely, the only intra-package import edges are:

| From (upper)               | imports | To (lower)   |
| -------------------------- | ------- | ------------ |
| `corrections`              | ──▶     | `psr`        |
| `psr`                      | ──▶     | `quadrature` |
| `generators`               | ──▶     | `linalg`     |

- **Base layer** (no intra-package imports): `linalg`, `quadrature`,
  `clustering`.
- **Middle layer**: `psr` (on `quadrature`), `generators` (on `linalg`).
- **Top layer**: `corrections` (on `psr`).

`clustering` is self-contained and depends only on third-party numerics
(`numpy`, `scipy`).

## Facades

Two `__init__.py` files re-export the public API and must stay consistent:

- `jsharpe.sharpe.__init__` re-exports every public symbol from the
  sub-modules, so `from jsharpe.sharpe import sharpe_ratio_variance` keeps
  working after the split into topical modules.
- `jsharpe.__init__` re-exports the **identical** set of symbols as the
  top-level public API.

Internal helpers (e.g. `quadrature.moments_Mk`, `quadrature.E_under_normal`)
are *not* part of the public surface and are reachable only via their full
module path.

## Enforcement

The layering is verified mechanically by an import-lint guard test
(`test_layering_is_acyclic_and_downward` in
`tests/jsharpe/sharpe/test_linalg.py`): it parses each module's intra-package
imports and fails if any lower layer imports an upper layer. The facade
consistency is guarded by `test_public_api_facades_are_consistent` in
`tests/jsharpe/sharpe/test_quadrature.py`.
