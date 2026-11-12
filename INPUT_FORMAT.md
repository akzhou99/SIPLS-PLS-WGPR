# Input data format

The workbook uses a fixed ordering for the well identifier, depth, and two ECS net spectra, followed by a **study-dependent number of mineral targets**. The number of minerals is **not fixed at eight**.

## Labeled workbook — 1,030 + K columns

Here, **K** is the number of mineral components selected for the study area (for example, K = 3, 4, or 8).

| Column positions | Content | Number of columns |
|---|---|---:|
| 1 | Well name | 1 |
| 2 | Depth | 1 |
| 3–516 | Net inelastic spectrum | 514 |
| 517–1030 | Net capture spectrum | 514 |
| 1031–(1030 + K) | Mineral contents | K |

Examples: **3 minerals = 1,033 columns; 4 minerals = 1,034 columns; 8 minerals = 1,038 columns.**

Each mineral target occupies one column, with the **mineral name as its column header** (e.g., `Quartz`, `Calcite`, `Clay`). Choose the mineral targets according to the study area. Every row represents one depth sample; the first row contains column headers.

## Prediction-only workbook — 1,030 columns

| Column positions | Content | Number of columns |
|---|---|---:|
| 1 | Well name | 1 |
| 2 | Depth | 1 |
| 3–516 | Net inelastic spectrum | 514 |
| 517–1030 | Net capture spectrum | 514 |

Prediction or test worksheets may also include **K reference mineral columns** when measured labels are available and evaluation metrics are required. In that case, the mineral columns must match the training targets in **name, number, and order**.

## Requirements

- Keep the first **1,030 columns** in the specified order. Do not add Excel index or other metadata columns.
- Select **K mineral targets** from the geology of your study area. Use the same K targets, column names, and order throughout the labeled datasets for one trained model.
- Mineral contents must be numeric and nonnegative; labeled rows used for training or evaluation must have values for all K targets.
- Spectral channels and depth must be numeric and contain no missing values. Well names must be present for every row.
- Training, test, and external-validation wells must not overlap.
- The default spectral order is **net inelastic, then net capture**. If using the reverse order, set `SPECTRA_ORDER = "capture-first"` in `config.py`.
- The supplied example workbook is **synthetic demonstration data**, not field measurements. Replace it with your own appropriately prepared ECS and mineral-composition data for research use.

**Implementation compatibility:** This document specifies the intended flexible target layout. If your current version of the Python scripts assumes exactly eight mineral columns, update the reader and any downstream target-count checks before using a dataset with K other than eight. Updating this document alone does not change the code.
