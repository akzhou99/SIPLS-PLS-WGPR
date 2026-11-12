# Example Data and Quick Test

## Purpose

The Excel workbook in this folder is provided solely to illustrate the required data structure and allow users to try the SIPLS–PLS–SSA-WGPR workflow.

> **Synthetic data only.** `Input data.xlsx` was generated artificially for software demonstration. It contains **no real well measurements**, **no measured ECS gamma-ray spectra**, and **no XRD mineral-composition data** from the manuscript. The simulated spectra are not intended to be a physically validated ECS simulator. Model accuracy or predictions obtained with this workbook must **not** be reported as scientific results or as a reproduction of the manuscript's experiments.

**For research or application, replace this example workbook with your own appropriately prepared ECS spectra and measured mineral contents.**

## Example workbook

File: [`Input data.xlsx`](Input%20data.xlsx)

| Excel sheet | Wells | Samples | Purpose |
|---|---:|---:|---|
| `Train well` | 6 synthetic wells (`SYN-TR-01`–`SYN-TR-06`) | 84 | Example training data |
| `Test well` | 2 synthetic wells (`SYN-TE-01`–`SYN-TE-02`) | 24 | Example held-out test data |
| `External well` | 2 synthetic wells (`SYN-EX-01`–`SYN-EX-02`) | 24 | Example independent-well validation data |

The three groups use **disjoint well names**. Each row represents a separate depth sample. All three sheets contain mineral labels so that prediction-error metrics can be calculated when supported by the scripts.

## Excel column layout

The first **1,030 columns** are fixed. Labeled workbooks then contain **K** mineral-composition columns, where K depends on the study area.

| Column positions (1-based) | Contents | Count |
|---|---|---:|
| 1 | Well name | 1 |
| 2 | Depth | 1 |
| 3–516 | Net inelastic spectrum | 514 |
| 517–1,030 | Net capture spectrum | 514 |
| 1,031–(1,030 + K) | Mineral composition | K |

The supplied example uses **K = 8**, or **1,038 columns** per sheet, with these illustrative mineral names: `Quartz`, `Feldspar`, `Calcite`, `Dolomite`, `Clay`, `Pyrite`, `Anhydrite`, and `Siderite`. **Eight is not a universal mineral requirement.** A different study may have 3, 4, or another number of mineral targets.

For prediction-only data without measured mineral labels, use only the first **1,030 columns**. Refer to the repository's [`INPUT_FORMAT.md`](../INPUT_FORMAT.md) for the general input specification.

**Code compatibility:** This example deliberately contains eight target minerals to match the original eight-mineral implementation. If your copy of the code still assumes eight targets, it must be adapted before using a dataset with another K; changing the Markdown documentation alone does not change model behavior.

## Quick Test

Run the following commands from the **repository root directory**.

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

### 2. Point the scripts to the example workbook

In the repository root `config.py`, configure the workbook path and the sheets as follows. Keep the other model settings from your existing `config.py` unchanged.

```python
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "example" / "Input data.xlsx"

TRAIN_SHEETS = ["Train well"]
TEST_SHEETS = ["Test well"]
EXTERNAL_SHEETS = ["External well"]

TRAIN_WELLS = []
TEST_WELLS = []
EXTERNAL_WELLS = []

SPECTRA_ORDER = "inelastic-first"
```

The well lists are empty because the three worksheets are already separated by well group. Do not mix these example wells with real wells in a scientific evaluation.

### 3. Train the model

```bash
python train.py
```

### 4. Predict held-out wells

After a fitted model has been saved, use the configured prediction commands:

```bash
python predict.py --split test
python predict.py --split external
```

**Runtime note:** The complete SIPLS–PLS–SSA–WGPR training process, especially SSA optimization, can be computationally intensive even on this small example. For an initial software smoke test, you may temporarily reduce the SSA population and iterations in `config.py`; such settings are **not equivalent to the manuscript's full experimental settings**.

## Use your own data

1. Replace `example/Input data.xlsx` with your own Excel workbook, or change `DATA_FILE` to its location.
2. Retain the first 1,030 columns in the specified order: well name, depth, 514 net-inelastic channels, then 514 net-capture channels.
3. Append **K** measured mineral-composition columns, using the mineral species appropriate to your study area. Mineral column names and order must match across labeled sheets.
4. Modify `TRAIN_SHEETS`, `TEST_SHEETS`, `EXTERNAL_SHEETS`, or the well-based partitions in `config.py` to reflect your own independent wells.
5. Confirm that the test/external-validation data are not used in training, feature selection, or hyperparameter optimization.
6. Ensure that the target-count handling in your Python code supports your chosen **K**. The original code may still require eight minerals.

**Privacy:** Do not publish proprietary real well names, depth measurements, spectra, or XRD results without permission. This repository's example workbook contains only simulated identifiers and data.

## Adding this folder to GitHub

Upload **both** `Input data.xlsx` and this `README.md` inside the repository's `example/` directory. If uploading through Git rather than the GitHub website, check `.gitignore`: a blanket `*.xlsx` rule will prevent the example workbook from being added unless it is explicitly allowed (or force-added with care).
