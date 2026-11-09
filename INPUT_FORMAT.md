# Input data format

The project uses one fixed Excel column layout. Please prepare the workbook in this format; the reader does not try to infer alternative layouts.

## Training workbook — 1,038 columns

| Column positions | Content | Number of columns |
|---|---|---:|
| 1 | Well name | 1 |
| 2 | Depth | 1 |
| 3–516 | Net inelastic spectrum | 514 |
| 517–1030 | Net capture spectrum | 514 |
| 1031–1038 | Mineral contents | 8 |

The eight mineral columns should use consistent names and order in every worksheet. Each row represents one depth sample. Do not add an Excel index column. The first row contains column headers.

## Prediction workbook — 1,030 columns

| Column positions | Content | Number of columns |
|---|---|---:|
| 1 | Well name | 1 |
| 2 | Depth | 1 |
| 3–516 | Net inelastic spectrum | 514 |
| 517–1030 | Net capture spectrum | 514 |

Prediction data may also contain the final eight mineral columns when reference measurements are available. If present, all eight values in a row must be filled to calculate evaluation metrics.

## Important

- The default spectral order is **net inelastic first, then net capture**. If the workbook uses the reverse order, change `SPECTRA_ORDER` in `config.py` to `"capture-first"`.
- Well names must be filled in on every row and must match the names in `TRAIN_WELLS`, `TEST_WELLS`, and `EXTERNAL_WELLS`.
- Training, test, and external-validation wells must be disjoint.
- The training workbook must contain eight numeric, nonnegative mineral contents per sample.
- Spectral channels and depth must be numeric and contain no blank values.
- Do not include an extra index column or other metadata columns.
