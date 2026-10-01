# Analysis Experiment Directory

In addition to the main paper experiments, TongBench contains several controlled analysis
experiments.

These experiments live under `analyze/`, alongside the following top-level directories:

```text
TongBench-dev/
  public/
  For_user/
  analyze/
```

## Organization

Create one directory under `analyze/` for each analysis experiment. The directory name should
identify the experiment, for example:

```text
analyze/
  <analysis_name>/
```

Each analysis directory contains the material specific to that experiment:

1. input
2. output
3. experiment-specific scripts
4. experiment-specific implementation

This keeps each experiment's inputs, outputs, and scripts together and prevents unrelated
experiments from becoming mixed together.

## Code Reuse

Analysis experiments should call the existing mainline implementation instead of copying or
reimplementing functionality that already exists.

Place a change in the corresponding analysis directory when it is specific to one experiment.
Change shared mainline code only when the capability is meant to be shared by all experiments.

## Boundary with For_user

`For_user/` contains handoff material for other purposes. Analysis experiments are kept under
`analyze/` and should not modify files under `For_user/`.
