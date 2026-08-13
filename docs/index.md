# H → μμ

This is the documentation for the **H→μμ** analysis — the search for the Higgs boson decaying to a
muon pair, built on the [FLAF framework](https://cms-flaf.github.io/FLAF/). It is a **single-Higgs**
analysis with the **leanest** FLAF setup.

!!! abstract "The common workflow lives in the FLAF docs"
    H→μμ runs on FLAF, so installation, the task pipeline (NanoAOD → anaTuples → histograms →
    plots), the configuration system, storage, eras and CI are **shared with every FLAF analysis**
    and documented once, in the **[FLAF documentation](https://cms-flaf.github.io/FLAF/)**:

    - [Prerequisites & installation](https://cms-flaf.github.io/FLAF/getting-started/installation/)
    - [Your first run](https://cms-flaf.github.io/FLAF/getting-started/first-run/)
    - [Full workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)
    - [Command arguments](https://cms-flaf.github.io/FLAF/workflow/arguments/)

    **This site covers only what is specific to H→μμ.**

## How H→μμ differs from the HH analyses

| Aspect | H→μμ |
|---|---|
| Process | **Single** Higgs (H→μμ), not di-Higgs. |
| Submodules | The simplest set: just **`FLAF`** and **`Corrections`**. |
| Statistical inference | **None** in this repository — there is no `StatInference`/`inference` submodule. |
| Eras | Runs over **all** Run 3 eras (CI lists them explicitly). |
| CI process names | **lower-case** (`custom_CI_signal`, `custom_CI_background`, `custom_CI_data`). |

## Quickstart

```sh
git clone --recursive git@github.com:cms-flaf/H_mumu.git
cd H_mumu
source env.sh                                  # first time builds the environment
voms-proxy-init -voms cms -rfc -valid 192:00
law index --verbose
```

Then smoke-test the chain (see
[FLAF → first run](https://cms-flaf.github.io/FLAF/getting-started/first-run/)):

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version my_first_run --period Run3_2022 --workflow local --branches 0 --test 1000
```

New to FLAF? Read [Key terms](https://cms-flaf.github.io/FLAF/getting-started/key-terms/) and
[Concepts](https://cms-flaf.github.io/FLAF/concepts/architecture/) first.

## Eras

H→μμ runs over all Run 3 eras (`Run3_2022`, `Run3_2022EE`, `Run3_2023`, `Run3_2023BPix`, and newer
as they come online). See [FLAF → Eras](https://cms-flaf.github.io/FLAF/concepts/eras/).
