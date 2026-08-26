# Running the analysis

The pipeline — anaTuples, observables, histograms, plots — is the **standard FLAF chain**; follow
the **[FLAF full-workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)** for
the commands (`InputFileTask` → … → `HistPlotTask`). This page collects what is **specific to
H→μμ**.

```sh
ERA=Run3_2022
VER=dev

law run FLAF.Analysis.tasks.HistPlotTask \
  --period $ERA --version $VER --workflow local --branches 0 --test 1000
```

## No statistical-inference stage here

Unlike the HH analyses, H→μμ does **not** include a statistical-inference step in this repository —
there is no `StatInference`/`inference` submodule, and the pipeline ends at histograms/plots. Any
interpretation is done with separate tooling outside this repository.

## Categories

H→μμ is split into production-mode categories (e.g. VBF- and ggH-enriched selections). Category and
channel selection is driven by `config/global.yaml`; adjust it there or via your
[`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/).

## Running all eras

H→μμ targets every Run 3 era. Run a stage per era, or — in CI — list them in `H_mumu_eras` (see
[FLAF → Integration pipeline](https://cms-flaf.github.io/FLAF/ci/integration-pipeline/)). Remember
that one `law run` processes **one** `--period` at a time.

## Choosing which variables to histogram

As for any FLAF analysis, the variable set is controlled by the `variables:` list in
`user_custom.yaml` (or `--variables`):

```yaml
variables:
  - mu1_pt
  - m_mumu
```

!!! note "Lower-case CI process names"
    H→μμ's CI process names are lower-case (`custom_CI_signal`, `custom_CI_background_TT`,
    `custom_CI_background_DY`, `custom_CI_data`) — unlike the capitalised names in the HH
    analyses. Use the exact name from `config/processes.yaml` when passing `--process`.

    There are two CI backgrounds: `custom_CI_background_TT` is one t̄t dataset (unstitched, like
    the real `TT` process) and `custom_CI_background_DY` is one DY dataset carrying the same
    stitcher the era's DY process uses — `DYMllStitcher` for 2022–2023BPix, the plain
    single-flavour `MCStitcher` for 2024 onwards. That is what runs the stitching over the
    whole anaTuple → merge chain in CI; keep them in step with the real processes.
