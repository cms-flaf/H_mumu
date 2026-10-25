# Running the analysis

The pipeline — anaTuples, observables, histograms, plots — is the **standard FLAF chain**; follow
the **[FLAF full-workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)** for
the commands (`InputFileTask` → … → `HistPlotTask`). This page collects what is **specific to
H→μμ**.

```sh
ERA=Run3_2022
VER=dev

law run FLAF.Analysis.tasks.HistPlotTask \
  --era $ERA --version $VER --workflow local --branches 0 --test 1000
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

H→μμ targets every Run 3 era. One `law run` covers several eras with `--era Run3_2022,Run3_2022EE,...`
(see [FLAF → Eras](https://cms-flaf.github.io/FLAF/concepts/eras/#running-several-eras)), or run a stage
per era; in CI, list them in `H_mumu_eras` (see
[FLAF → Integration pipeline](https://cms-flaf.github.io/FLAF/ci/integration-pipeline/)).

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

## Blinded mass window

The plots of `m_mumu` and its variants (the `m_mumu(_.*)?` entry of `config/plot/histograms.yaml`)
show no data between 115 and 135 GeV, in every region: `blind_ranges: [ 115, 135 ]` hides the data
points of every bin that overlaps the window, in the plot and in the ratio panel. It acts on the
`m_mumu` axis only: events inside the window still enter the plots of the other variables in the
regions that select them (`Signal_Fit`, `Signal_ext`, `mass_inclusive`), the NN outputs included,
unless those variables are blinded themselves (`blind: true`). The key is
described under
[FLAF → HistPlotTask](https://cms-flaf.github.io/FLAF/reference/tasks/#histplottask).
