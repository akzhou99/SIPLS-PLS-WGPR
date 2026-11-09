"""Train SIPLS–PLS–SSA-WGPR on the selected training wells."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from models.pipeline import SIPLSPLSSSAWGPR
from models.wgpr import WeightedGPR
from models.data import check_splits, check_well_splits, read_workbook, subset_wells


def load_partitions(input_file, train_sheets, test_sheets, external_sheets,
                    train_wells, test_wells, external_wells):
    well_mode = bool(train_wells or test_wells or external_wells)
    if well_mode:
        check_well_splits(train_wells, test_wells, external_wells)
        input_sheets = list(dict.fromkeys(train_sheets + test_sheets + external_sheets))
        if not input_sheets:
            raise ValueError("Specify the worksheets containing the 10 wells")
        data = read_workbook(input_file, input_sheets, order=config.SPECTRA_ORDER,
                             require_labels=False)
        if data.well_id is None:
            raise ValueError("Well-based splitting requires a well-name column")
        assigned = set(train_wells + test_wells + external_wells)
        available = set(data.well_id)
        unassigned = available - assigned
        missing = assigned - available
        if unassigned or missing:
            raise ValueError(f"Well assignment does not match the input workbook. "
                             f"Unassigned={sorted(unassigned)}, missing={sorted(missing)}")
        train_data = subset_wells(data, train_wells)
        grouped_wells = {"train": set(train_wells), "test": set(test_wells),
                         "external": set(external_wells)}
        print(f"Well split verified against input workbook: "
              f"{tuple(map(len, grouped_wells.values()))}")
    else:
        check_splits(train_sheets, test_sheets, external_sheets)
        input_sheets = list(dict.fromkeys(train_sheets + test_sheets + external_sheets))
        train_data = read_workbook(input_file, train_sheets,
                                   order=config.SPECTRA_ORDER, require_labels=True)
        grouped_wells = {
            "train": set(train_data.well_id) if train_data.well_id is not None else None
        }
        for part, sheets in (("test", test_sheets), ("external", external_sheets)):
            if sheets:
                other = read_workbook(input_file, sheets, order=config.SPECTRA_ORDER,
                                      require_labels=False)
                grouped_wells[part] = set(other.well_id) if other.well_id is not None else None
        if train_data.well_id is not None and len(grouped_wells["train"]) > 6:
            raise ValueError("Training worksheets contain more than 6 wells. "
                             "Set TRAIN_WELLS, TEST_WELLS and EXTERNAL_WELLS "
                             "to split the combined workbook by well name.")
        # Check every available pair, even when one held-out partition has not
        # yet been configured. Previously an empty external partition skipped the
        # train-vs-test overlap check completely.
        for left, right in (("train", "test"), ("train", "external"),
                            ("test", "external")):
            if grouped_wells.get(left) is not None and grouped_wells.get(right) is not None:
                if grouped_wells[left] & grouped_wells[right]:
                    raise ValueError(f"Well-name overlap between {left} and {right}")
        if all(part in grouped_wells and grouped_wells[part] is not None
               for part in ("train", "test", "external")):
            counts = tuple(len(grouped_wells[k]) for k in ("train", "test", "external"))
            if counts != (6, 2, 2):
                raise ValueError(f"Expected 6/2/2 wells in the paper, got {counts}")
            print(f"Well split verified: {counts}")

    if train_data.y is None or not np.isfinite(train_data.y).all():
        raise ValueError("Training wells must have 8 finite mineral columns")
    return train_data, grouped_wells, input_sheets, well_mode


def train(input_file, output_dir, train_sheets, test_sheets, external_sheets, evaluator, cv, pls_cv,
          train_wells=None, test_wells=None, external_wells=None,
          ssa_population=None, ssa_iterations=None):
    train_wells = train_wells or []
    test_wells = test_wells or []
    external_wells = external_wells or []
    train_data, grouped_wells, input_sheets, well_mode = load_partitions(
        input_file, train_sheets, test_sheets, external_sheets,
        train_wells, test_wells, external_wells)

    print(f"Training samples: {len(train_data.X)}")
    print(f"Training wells: {sorted(grouped_wells['train']) if grouped_wells['train'] else 'unspecified'}")
    print("Feature selection CV: shuffled KFold on training samples; no leave-one-well-out CV")
    gpr = WeightedGPR(
        seed=config.RANDOM_STATE,
        population=config.SSA_POPULATION if ssa_population is None else ssa_population,
        iterations=config.SSA_ITERATIONS if ssa_iterations is None else ssa_iterations,
        discoverer_ratio=config.SSA_DISCOVERER_RATIO,
        warning_threshold=config.SSA_WARNING_THRESHOLD,
        cv_folds=config.WGPR_CV_FOLDS,
        alpha=config.WGPR_ALPHA,
    )
    model = SIPLSPLSSSAWGPR(
        schemes=tuple(config.SIPLS_SCHEMES),
        sipls_components=config.SIPLS_COMPONENTS,
        pls_max_components=config.PLS_MAX_COMPONENTS,
        evaluator=evaluator, cv=cv, pls_cv=pls_cv, spectrum_scale=config.SPECTRUM_SCALE,
        seed=config.RANDOM_STATE, wgpr=gpr,
    ).fit(train_data.X, train_data.y)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.joblib")
    names = train_data.mineral_names or [f"Mineral_{i + 1}" for i in range(8)]
    model.wgpr.parameter_table(names).to_csv(output_dir / "wgpr_parameters.csv", index=False)
    results = []
    for kind, selector in (("inelastic", model.sipls_i_), ("capture", model.sipls_c_)):
        for row in selector.results_:
            results.append({"spectrum": kind,
                            **{k: v for k, v in row.items() if k != "selected_intervals"},
                            "selected_intervals": str(row["selected_intervals"])})
    pd.DataFrame(results).to_csv(output_dir / "sipls_results.csv", index=False)
    model.pls_.cv_results_.to_csv(output_dir / "pls_cv.csv", index=False)
    weights = model.pls_.weights_
    pd.DataFrame({"component": np.arange(1, len(weights) + 1),
                  "weight": weights, "cumulative_weight": np.cumsum(weights)}).to_csv(
                      output_dir / "pls_weights.csv", index=False)
    split_info = {
        "split_mode": "well" if well_mode else "sheet",
        "train_sheets": train_sheets, "test_sheets": test_sheets,
        "external_sheets": external_sheets, "input_sheets": input_sheets,
        "train_wells": train_wells, "test_wells": test_wells,
        "external_wells": external_wells, "input_workbook": str(input_file),
        "mineral_names": names, "spectra_order": config.SPECTRA_ORDER,
        "evaluator": evaluator, "cv": cv, "pls_cv": pls_cv,
        "cv_type": "sample KFold within training wells",
        "n_training_samples": len(train_data.X),
        "well_counts": {k: len(v) if v is not None else None
                        for k, v in grouped_wells.items()},
        "n_pls_components": model.pls_.n_components_,
        "n_inelastic_channels": len(model.sipls_i_.indices_),
        "n_capture_channels": len(model.sipls_c_.indices_),
        "sipls_schemes": config.SIPLS_SCHEMES,
        "sipls_pls_components": config.SIPLS_COMPONENTS,
        "pls_max_components": config.PLS_MAX_COMPONENTS,
        "spectrum_scale": config.SPECTRUM_SCALE,
        "ssa_population": gpr.population,
        "ssa_iterations": gpr.iterations,
        "wgpr_cv_folds": gpr.cv_folds,
        "wgpr_kernel_candidates": list(gpr.kernel_names),
        "wgpr_length_bounds": gpr.length_bounds,
        "wgpr_signal_bounds": gpr.signal_bounds,
        "wgpr_noise_bounds": gpr.noise_bounds,
    }
    (output_dir / "split.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved trained model and selection tables to {output_dir.resolve()}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.DATA_FILE)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--train-sheets", nargs="+", default=config.TRAIN_SHEETS)
    parser.add_argument("--test-sheets", nargs="*", default=config.TEST_SHEETS)
    parser.add_argument("--external-sheets", nargs="*", default=config.EXTERNAL_SHEETS)
    parser.add_argument("--train-wells", nargs="*", default=config.TRAIN_WELLS)
    parser.add_argument("--test-wells", nargs="*", default=config.TEST_WELLS)
    parser.add_argument("--external-wells", nargs="*", default=config.EXTERNAL_WELLS)
    parser.add_argument("--evaluator", choices=("xgboost", "pls"), default=config.EVALUATOR)
    parser.add_argument("--cv", type=int, default=config.CV_FOLDS, help="SIPLS cross-validation folds")
    parser.add_argument("--pls-cv", type=int, default=config.PLS_CV_FOLDS, help="PLS component-selection folds")
    parser.add_argument("--ssa-population", type=int, default=config.SSA_POPULATION,
                        help="SSA population size (30 for manuscript settings)")
    parser.add_argument("--ssa-iterations", type=int, default=config.SSA_ITERATIONS,
                        help="SSA iteration count (50 for manuscript settings)")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate sheets and well partitions without fitting a model")
    args = parser.parse_args()
    if args.check_data:
        data, groups, _, _ = load_partitions(
            args.input, args.train_sheets, args.test_sheets, args.external_sheets,
            args.train_wells, args.test_wells, args.external_wells)
        print(f"Validated {len(data.X)} training rows, {data.X.shape[1]} channels, "
              f"{data.y.shape[1]} targets. Training wells: "
              f"{sorted(groups['train']) if groups['train'] else 'unspecified'}")
        return
    train(args.input, args.output, args.train_sheets, args.test_sheets,
          args.external_sheets, args.evaluator, args.cv, args.pls_cv,
          args.train_wells, args.test_wells, args.external_wells,
          args.ssa_population, args.ssa_iterations)


if __name__ == "__main__":
    main()
