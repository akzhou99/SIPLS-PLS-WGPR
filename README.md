# SIPLS–PLS–SSA-WGPR

A modular implementation of the ECS spectral workflow for quantitative mineral prediction.

## Workflow

1. Calculate net inelastic and net capture spectra and align ECS spectra with XRD depths.
2. Preprocess both spectra, select spectral intervals using SIPLS, and extract PLS latent variables.
3. Calculate PLS variance weights and train mineral-specific WGPR models. SSA optimizes the kernel hyperparameters.
4. Predict held-out wells and continuous-depth ECS data; export predictions and uncertainty estimates.

## Project layout

```text
.
├── config.py
├── prepare_data.py
├── train.py
├── predict.py
├── models/
│   ├── spectra.py       # Net-spectrum calculation and ECS–XRD alignment
│   ├── data.py          # Excel reading and spectral preprocessing
│   ├── pipeline.py
│   ├── sipls.py
│   ├── pls.py
│   ├── wgpr.py
│   ├── ssa.py
│   └── _cv.py
├── INPUT_FORMAT.md
├── requirements.txt
└── .gitignore
```

## Input workbook

Use the fixed Excel column layout described in [INPUT_FORMAT.md](INPUT_FORMAT.md). Training workbooks contain 1,038 columns; prediction-only workbooks contain 1,030 columns. The first column is the well name, the second is depth, followed by 514 net-inelastic channels, 514 net-capture channels, and (for training) eight mineral contents.

## Configuration

Edit `config.py` before training:

- `TRAIN_SHEETS`, `TEST_SHEETS`, `EXTERNAL_SHEETS`: worksheet-based split; or
- `TRAIN_WELLS`, `TEST_WELLS`, `EXTERNAL_WELLS`: explicit well-name split for a combined worksheet.

For the manuscript split, assign six training wells, two test wells, and two independent external-validation wells. The three well lists must be disjoint.

The default settings use PLS for SIPLS interval evaluation, 10-fold sample-level CV for SIPLS, five-fold CV for PLS component selection, and the four WGPR kernel candidates described in the manuscript. SSA defaults to a population of 30 and 50 iterations.

## Installation

```bash
pip install -r requirements.txt
```

## Run

Validate the workbook and split configuration:

```bash
python train.py --check-data
```

Train and save the model:

```bash
python train.py
```

Predict the configured test wells or independent external-validation wells:

```bash
python predict.py --split test
python predict.py --split external
```

Prepare net spectra or align ECS spectra to XRD depths when starting from raw source workbooks:

```bash
python prepare_data.py net --input raw_ecs.xlsx --output net_spectra.xlsx
python prepare_data.py align --ecs-file net_spectra.xlsx --xrd-file xrd.xlsx --output aligned.xlsx
```

## Outputs

The training output directory contains the fitted model, SIPLS selection results, PLS cross-validation results, latent-variable weights, WGPR parameter table, and the recorded data split. Prediction outputs contain depth, well identifier (when available), mineral predictions, predictive standard deviations, and evaluation metrics when reference labels are present.

All selection and model parameters are estimated from the training wells. Test and external-validation wells are transformed with the fitted training pipeline and are not used to fit the model.

PLS solver settings: `max_iter=2000`, `tol=1e-6` (shared by CV and final PLS fitting).

## Raw ECS worksheets

The `net` command expects four headerless worksheets named `SBUR`, `SEAR`, `SLAT`, and `STAU`, each with depth followed by 514 channels. The output uses `Well_ID` (when provided), `Depth`, 514 net-inelastic channels, and 514 net-capture channels. For the training workbook, place the well name first, depth second, and append eight mineral columns.

The `align` command expects an XRD worksheet named `XRD` by default (override with `--xrd-sheet`).

The default training workbook is `data/Input data.xlsx`; set `ECS_DATA_FILE` or edit `DATA_FILE` in `config.py` to use another path.
