# Setup

H→μμ is installed and run like any FLAF analysis, and it has the **simplest** setup of the three —
only the shared `FLAF` and `Corrections` submodules. The general procedure (prerequisites, what the
first `source env.sh` builds) is in the
**[FLAF installation guide](https://cms-flaf.github.io/FLAF/getting-started/installation/)**.

## Clone

```sh
git clone --recursive git@github.com:cms-flaf/H_mumu.git
cd H_mumu
source env.sh
```

`--recursive` pulls `FLAF` and `Corrections`; without it, imports fail on empty directories.

!!! tip "Working from a fork"
    For contributions it is convenient to
    [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
    `cms-flaf/H_mumu` and add your fork as a remote. See
    [FLAF → Contributing](https://cms-flaf.github.io/FLAF/contributing/).

## Submodules

| Submodule | Role |
|---|---|
| `FLAF` | The shared framework. |
| `Corrections` | Object corrections & systematics. |

There are **no** analysis-specific physics submodules and **no** statistical-inference submodule —
this is what makes H→μμ the lightest analysis to set up.

## Production model

The production [physics model](https://cms-flaf.github.io/FLAF/configuration/processes-and-models/)
is `BaseModel` (set in `config/global.yaml`). For fast local tests, set `phys_model: TestModel` in
your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/).

## Next

- [Running the analysis](analysis.md) — the H→μμ-specific run notes.
- [FLAF → Full workflow](https://cms-flaf.github.io/FLAF/workflow/walkthrough/) — the common
  pipeline, stage by stage.
