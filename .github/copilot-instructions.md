# H_mumu — instructions for Copilot code review

The H→μμ analysis, and the leanest member of the [FLAF](https://github.com/cms-flaf/FLAF)
ecosystem: single-Higgs rather than di-Higgs, two submodules, no statistical-inference chain.

**Read `FLAF/.github/copilot-instructions.md` first.** It carries the framework invariants — law
task semantics, bundles, remote-storage caching, processor stages, concurrency — and the rules on
what a useful comment looks like and what not to flag. The rule that documentation ships in the same PR applies here too, and is restated below with the pages that matter for this repository. Everything there applies here. This file
adds only what is specific to this analysis.

## Analysis-specific invariants

### Naming

**The CI process names are lower-case here** (`custom_CI_signal`, `custom_CI_background_TT`,
`custom_CI_background_DY`, `custom_CI_data`) where the HH analyses capitalise them. Copying a
snippet from HH_bbtautau or HH_bbWW without adjusting the case silently selects nothing.

### Stitching

- DY is stitched with `DYMllStitcher` (`*DY_flavor_mll_processors`) for 2022–2023BPix and with the
  plain single-flavour `MCStitcher` from 2024 onwards; the per-era difference is deliberate.
- `DYMllStitcher` derives `LHE_dilep_flavor` and `LHE_mll` from `LHEPart`, which this anaTuple
  **does** store (`AnaProd/anaTupleDef.py` requests the `LHE` and `LHEPart` observable groups, defined in `AnaProd/observables.py`). A change that stops
  storing those groups breaks the merge-stage stitching, with no failure until then.
- Every stitching processor must declare `stages: [ AnaTuple, AnaTupleMerge ]` — see the FLAF file
  for what happens otherwise.

### Integration test

`TestModel` runs `custom_CI_background_TT` and `custom_CI_background_DY` plus one signal and one
data process, and each must carry the same `processors:` as the real `TT` / DY process **of that
era**. A diff that changes a real process's processors and leaves the CI counterpart behind
silently removes the coverage.

The process names are also listed in `cms-flaf/FLAF_ci`, a **different repository**; renaming or
adding one here needs that updated in step.

## Documentation must ship with the change

A PR must update the documentation **in the same PR** whenever it changes anything a user of the
framework can observe. Treat this as a review item of the same weight as correctness — docs
drifting from the code is the failure that motivated the current documentation, and a PR that
lands without them is not complete.

Ask, for every diff: does it add, rename or remove any of these?

- a task or DAG node, or the arguments/parameters of one;
- a command, a CLI flag, or the meaning of an existing one;
- a configuration key — `global.yaml`, `user_custom.yaml`, `processes.yaml`, `phys_models.yaml`,
  cross-sections, `fs_*` storage keys, bundle flavours, processor entries;
- a dataset, era, process or physics-model name;
- the environment, installation or setup steps;
- storage locations, output paths or log locations;
- a CI workflow, or how the integration test is triggered or configured;
- any behaviour a user relies on, including a default that changes.

If the answer is yes and the diff touches **no** documentation file, say so and name the page that
should have changed. If the author states the change is internal-only, that is a legitimate
answer — a pure refactor or bugfix with no user-visible effect is exempt — but it should be
stated in the PR, not left implicit.

Also flag the inverse: documentation edited to describe behaviour the diff does not implement, and
new pages added without being wired into `mkdocs.yml`'s `nav` (the build fails on that, but the
review should catch it first).

Where it goes:

- `docs/` in this repository for analysis-specific material (`analysis.md`, `setup.md`, `index.md`).
- **`FLAF/docs/` for anything framework-wide.** If the change alters shared behaviour, the
  documentation belongs there, in a companion PR to `cms-flaf/FLAF` — flag that it is missing
  rather than accepting an analysis-local description of a framework change.
- New pages must be added to `nav:` in `mkdocs.yml`; verified with `mkdocs build --strict`.

## Repository facts

Verified 2026-08-27; re-check before relying on any of it.

| | |
|---|---|
| Layout | `AnaProd/` (`anaTupleDef.py`, `baseline.py`, `observables.py`), `Analysis/` (`H_mumu.py`, `histTupleDef.py`, ONNX models), `Studies/DNN/`, `config/`, `include/` (`Helper.h`, `HmumuCore.h`, `MuonScaRe.cc`), `docs/` |
| Submodules | `FLAF` and `Corrections` only — no `StatInference`, no `inference` |
| Eras | Run 3: 2022, 2022EE, 2023, 2023BPix, 2024, 2025, 2026 |
| Configs | `config/global.yaml`, `config/processes.yaml` (processor anchors), `config/phys_models.yaml`, `config/<era>/` |
| Tests | No unit tests in this repo; the framework's suites live in `FLAF/test/` |
| Workflows | `formatting-check`, `repo-sanity-checks`, `test-setup-loading`, `deploy-docs`, `trigger-flaf-integration`. Formatting and era loading are checked automatically — do not comment on them |
| Integration test | Triggered by `@cms-flaf-bot please test`; its configuration lives in `cms-flaf/FLAF_ci`, **not** in this repo |
| Docs | `docs/`, plus the shared framework docs in `FLAF/docs/` |
