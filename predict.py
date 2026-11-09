"""Predict test, external or new wells using a fitted model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from models.pipeline import SIPLSPLSSSAWGPR
from models.data import read_workbook, subset_wells


def predict(model_path, sheets, input_file, output_dir, *, order, mineral_names=None,
            well_names=None):
    model = SIPLSPLSSSAWGPR.load(model_path)
    data = read_workbook(input_file, sheets, order=order, require_labels=False)
    if well_names is not None:
        data = subset_wells(data, well_names)
    if data.mineral_names and mineral_names and data.mineral_names != mineral_names:
        raise ValueError("Mineral column names differ from those used for training")
    if data.y is not None and np.isfinite(data.y).any():
        sums = np.nansum(data.y, axis=1)
        labeled = np.isfinite(data.y).all(axis=1)
        if labeled.any():
            median_sum = float(np.median(sums[labeled]))
            if model.closed_composition_ and model.target_scale_ == 100.0 and abs(median_sum - 1) < 0.1:
                raise ValueError("Prediction labels appear to be fractions (0-1), but the training labels use percent (0-100)")
            if model.closed_composition_ and model.target_scale_ == 1.0 and abs(median_sum - 100) < 10:
                raise ValueError("Prediction labels appear to be percent (0-100), but the training labels use fractions/raw units")
    names = mineral_names or data.mineral_names or [f"Mineral_{i + 1}" for i in range(8)]
    predictions, stds = model.predict(data.X)
    result = pd.DataFrame({"Depth": data.depth, "Sheet": data.source_sheet})
    if data.well_id is not None:
        result.insert(0, "Well_ID", data.well_id)
    for index, name in enumerate(names):
        result[f"Pred_{name}"] = predictions[:, index]
        result[f"Std_{name}"] = stds[:, index]
        if data.y is not None:
            result[f"True_{name}"] = data.y[:, index]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    if data.y is not None:
        rows = []
        grouping = data.well_id if data.well_id is not None else data.source_sheet
        for group in sorted(set(grouping)):
            for index, name in enumerate(names):
                valid = (grouping == group) & np.isfinite(data.y[:, index])
                if not valid.any():
                    continue
                truth, pred = data.y[valid, index], predictions[valid, index]
                rows.append({"well_or_sheet": group, "mineral": name, "n": int(valid.sum()),
                             "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
                             "mae": float(mean_absolute_error(truth, pred)),
                             "r2": float(r2_score(truth, pred)) if valid.sum() >= 2 else np.nan})
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    print(f"Predictions written to {output_dir.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.OUTPUT_DIR / "model.joblib")
    parser.add_argument("--input", type=Path, help="Workbook containing test/new-well data")
    parser.add_argument("--split", choices=("test", "external", "custom"), default="test")
    parser.add_argument("--sheets", nargs="+", help="Worksheet override")
    parser.add_argument("--order", choices=("inelastic-first", "capture-first"))
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "prediction")
    args = parser.parse_args()
    split_path = args.model.parent / "split.json"
    if not split_path.exists():
        raise FileNotFoundError("Missing split.json; use the matching trained model folder")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    key = {"test": "test_sheets", "external": "external_sheets", "custom": None}[args.split]
    well_names = None
    if args.split == "custom":
        if not args.sheets:
            parser.error("--split custom requires --sheets")
        sheets = args.sheets
    elif split.get("split_mode") == "well":
        sheets = args.sheets or split["input_sheets"]
        well_names = split[f"{args.split}_wells"]
        if not well_names:
            parser.error(f"No {args.split} wells configured")
    else:
        sheets = args.sheets or split[key]
        if not sheets:
            parser.error("Configure the selected test/external worksheets before training")
        if set(sheets) & set(split["train_sheets"]):
            parser.error("Training worksheets cannot be used as held-out data")
    predict(args.model, sheets, args.input or split["input_workbook"], args.output,
            order=args.order or split["spectra_order"],
            mineral_names=split["mineral_names"], well_names=well_names)


if __name__ == "__main__":
    main()
