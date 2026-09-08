ASRC experiment code and benchmark datasets

Source and benchmark-data snapshot, 2026-09-08.

This private repository contains experiment code, run configurations, dependency
specifications, four benchmark datasets, and third-party attribution files. It keeps
the original relative directory layout, including historical experiment-specific
source snapshots and baseline adaptations.

Main entry points:
  reproduction/sota/train_complex.py       ASRC training implementation
  reproduction/sota/train_support.py       Support-experiment training
  reproduction/sota/evaluate_three_seed.py Repeated evaluation entry point
  reproduction/sota/aggregate_three_seed.py
  reproduction/sota/aggregate_support.py
  reproduction/strict_baselines/           Baseline adapters and protocol code
  reproduction/tests/                     Existing experiment checks

Other directories:
  data/raw/     Canonical DBP-5L, E-PKG, DWY, and WK3l-15k data
  data/manifests/ Dataset checksums, counts, and alignment inventories
  baselines/    Baseline source releases
  deployment/   Experiment launch and remote-execution scripts
  work/         Baseline working copies and preparation utilities
  outputs/      Analysis and table/figure-generation source scripts

Dependencies and configuration:
  Existing requirements files and configuration files remain beside their
  original entry points. The project uses multiple method-specific environments.
  Dataset paths and historical machine paths in configuration records are kept
  as recorded and must be adapted for another machine.

Included benchmark data:
  DBP-5L, E-PKG, and DWY use the recorded DMKGC release; DWY retains the DMKGC
  split. WK3l-15k includes both EN_F and FR plus their alignment links, from the
  recorded ATransN release. Original train/validation/test splits, entity and
  relation dictionaries, and alignment files are preserved byte for byte.
  data/SOURCE_PROVENANCE.json records the upstream URLs and source commits.
  reproduction/strict_baselines/data_manifests/ contains the 16 recorded input
  manifests, including validation-index selections used by the experiment code.

Excluded from this repository:
  Model weights, checkpoints, derived embeddings, graph caches, experiment results,
  manuscript files, papers, Markdown files, rendered figures, archives,
  installed Python environments, and caches.

Nine Jupyter notebooks contain only their code cells; stored execution outputs,
execution counts, and non-code cells were removed from these uploaded copies.
Other copied files are byte-identical to their source files.

SOURCE_MANIFEST.json lists each imported code/data file, its category, original and uploaded
SHA-256, and any notebook transformation. This upload did not run experiments.
Preserved baseline license/notice files apply to their respective code.
