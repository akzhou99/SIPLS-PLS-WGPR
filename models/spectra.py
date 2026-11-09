"""Compute net ECS spectra and align them with core XRD measurements."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


ECS_SHEETS = {
    "SBUR": "SBUR",
    "SEAR": "SEAR",
    "SLAT": "SLAT",
    "STAU": "STAU",
}

N_CHANNELS = 514
INELASTIC_COLUMNS = [f"NetInelastic_{i:03d}" for i in range(1, N_CHANNELS + 1)]
CAPTURE_COLUMNS = [f"NetCapture_{i:03d}" for i in range(1, N_CHANNELS + 1)]


def compute(
    input_path: str | Path,
    output_path: str | Path,
    *,
    well_id: str | None = None,
    missing_policy: str = "error",
    negative_policy: str = "clip",
) -> pd.DataFrame:
    """Calculate both net spectra from the four raw ECS worksheets."""
    if missing_policy not in {"error", "zero"}:
        raise ValueError("missing_policy must be 'error' or 'zero'")
    if negative_policy not in {"error", "clip"}:
        raise ValueError("negative_policy must be 'error' or 'clip'")

    spectra = {}
    for spectrum_name, sheet_name in ECS_SHEETS.items():
        frame = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
        if frame.shape[1] < N_CHANNELS + 1:
            raise ValueError(
                f"{sheet_name}: expected depth plus {N_CHANNELS} channels; "
                f"found {frame.shape[1]} columns"
            )

        depth = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(depth).all():
            raise ValueError(f"{sheet_name}: depth contains missing or non-numeric values")
        if len(np.unique(depth)) != len(depth):
            raise ValueError(f"{sheet_name}: duplicate depths are not allowed")

        channels = frame.iloc[:, 1 : N_CHANNELS + 1].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        invalid = ~np.isfinite(channels)
        if invalid.any():
            if missing_policy == "error":
                raise ValueError(
                    f"{sheet_name}: found {invalid.sum()} invalid channel values. "
                    "Check the source data or set missing_policy='zero'."
                )
            channels[invalid] = 0.0
            print(f"Warning: replaced {invalid.sum()} invalid values in {sheet_name} with zero")

        spectra[spectrum_name] = (depth, channels)

    reference_depth = spectra["SEAR"][0]
    for spectrum_name in ("SBUR", "SLAT", "STAU"):
        depth = spectra[spectrum_name][0]
        if depth.shape != reference_depth.shape or not np.allclose(
            depth, reference_depth, atol=1e-6, rtol=0
        ):
            raise ValueError(
                f"Depth rows in {spectrum_name} do not match SEAR. "
                "Align the source worksheets before calculating net spectra."
            )

    net_inelastic = spectra["SBUR"][1] - spectra["SEAR"][1]
    net_capture = spectra["SEAR"][1] + spectra["SLAT"][1] + spectra["STAU"][1]

    negative_inelastic = int(np.count_nonzero(net_inelastic < 0))
    negative_capture = int(np.count_nonzero(net_capture < 0))
    if negative_inelastic or negative_capture:
        if negative_policy == "error":
            raise ValueError(
                "Negative net counts found: "
                f"inelastic={negative_inelastic}, capture={negative_capture}"
            )
        print(
            "Warning: clipping negative net counts to zero "
            f"(inelastic={negative_inelastic}, capture={negative_capture})"
        )
        net_inelastic = np.clip(net_inelastic, 0.0, None)
        net_capture = np.clip(net_capture, 0.0, None)

    result = pd.DataFrame(
        np.column_stack((reference_depth, net_inelastic, net_capture)),
        columns=["Depth"] + INELASTIC_COLUMNS + CAPTURE_COLUMNS,
    )
    if well_id is not None:
        result.insert(1, "Well_ID", str(well_id))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output, index=False)
    print(
        f"Saved {output}: {len(result)} spectra, "
        "514 inelastic and 514 capture channels"
    )
    return result


def read_ecs(
    path: str | Path,
    sheet: str,
    *,
    no_header: bool = False,
    unlabelled_order: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Read a net-spectrum worksheet and return depth, spectra, and well ID."""
    if unlabelled_order not in {None, "inelastic-first", "capture-first"}:
        raise ValueError("unlabelled_order must be 'inelastic-first' or 'capture-first'")

    frame = pd.read_excel(path, sheet_name=sheet, header=None if no_header else 0)
    well_id = None

    well_columns = [column for column in frame.columns if str(column).strip() == "Well_ID"]
    if len(well_columns) > 1:
        raise ValueError("More than one well identifier column was found")
    if well_columns:
        well_values = frame[well_columns[0]].dropna().astype(str).str.strip().unique()
        if len(well_values) > 1:
            raise ValueError("A single alignment operation can process only one well")
        if len(well_values) == 1:
            well_id = str(well_values[0])

    if set(INELASTIC_COLUMNS + CAPTURE_COLUMNS).issubset(frame.columns):
        channel_frame = frame[INELASTIC_COLUMNS + CAPTURE_COLUMNS]
    else:
        if not no_header or unlabelled_order is None:
            raise ValueError(
                "Net-spectrum channel names were not found. Headered data must use "
                "the standard channel names; headerless data requires no_header=True "
                "and an explicit unlabelled_order."
            )
        expected_columns = 1 + 2 * N_CHANNELS
        if frame.shape[1] != expected_columns:
            raise ValueError(
                f"Headerless data must contain exactly {expected_columns} columns; "
                f"found {frame.shape[1]}"
            )
        channel_frame = frame.iloc[:, 1:expected_columns]
        if unlabelled_order == "capture-first":
            channel_frame = pd.concat(
                [channel_frame.iloc[:, N_CHANNELS:], channel_frame.iloc[:, :N_CHANNELS]],
                axis=1,
            )

    if "Depth" in frame.columns:
        depth_series = frame["Depth"]
    else:
        non_well_columns = [column for column in frame.columns if column not in well_columns]
        if not non_well_columns:
            raise ValueError("Depth column was not found")
        depth_series = frame[non_well_columns[0]]

    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    channels = channel_frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if channels.shape[1] != 2 * N_CHANNELS:
        raise ValueError(f"Expected {2 * N_CHANNELS} spectral channels; found {channels.shape[1]}")
    if not np.isfinite(depth).all() or not np.isfinite(channels).all():
        raise ValueError("Depth or spectral counts contain missing or non-finite values")

    return depth, channels, well_id


def align(
    *,
    ecs_file: str | Path,
    ecs_sheet: str,
    xrd_file: str | Path,
    xrd_sheet: str,
    output: str | Path,
    well_id: str | None = None,
    out_of_range: str = "error",
    duplicate_policy: str = "error",
    ecs_no_header: bool = False,
    unlabelled_order: str | None = None,
) -> pd.DataFrame:
    """Interpolate net ECS spectra to XRD depths and append mineral labels."""
    if out_of_range not in {"error", "skip"}:
        raise ValueError("out_of_range must be 'error' or 'skip'")
    if duplicate_policy not in {"error", "mean"}:
        raise ValueError("duplicate_policy must be 'error' or 'mean'")

    ecs_depth, channels, imported_well_id = read_ecs(
        ecs_file,
        ecs_sheet,
        no_header=ecs_no_header,
        unlabelled_order=unlabelled_order,
    )
    xrd = pd.read_excel(xrd_file, sheet_name=xrd_sheet)
    if xrd.shape[1] < 9:
        raise ValueError("XRD data must contain one depth column and eight mineral columns")

    xrd_depth = pd.to_numeric(xrd.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    minerals = xrd.iloc[:, 1:9].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(xrd_depth).all() or not np.isfinite(minerals.to_numpy(dtype=float)).all():
        raise ValueError("XRD depths or mineral contents contain missing or non-numeric values")
    if len(set(xrd.columns[1:9])) != 8:
        raise ValueError("The eight XRD mineral columns must have unique names")
    if well_id is not None and imported_well_id is not None and str(well_id) != imported_well_id:
        raise ValueError(f"Well ID mismatch: ECS={imported_well_id}, requested={well_id}")
    well_id = str(well_id) if well_id is not None else imported_well_id

    if len(ecs_depth) < 2:
        raise ValueError("At least two ECS depth points are required for interpolation")

    order = np.argsort(ecs_depth)
    ecs_depth = ecs_depth[order]
    channels = channels[order]
    if np.any(np.diff(ecs_depth) == 0):
        if duplicate_policy == "error":
            raise ValueError("Duplicate ECS depths found; use duplicate_policy='mean' if appropriate")
        grouped = pd.DataFrame(channels)
        grouped.insert(0, "Depth", ecs_depth)
        grouped = grouped.groupby("Depth", sort=True, as_index=False).mean()
        ecs_depth = grouped.iloc[:, 0].to_numpy(dtype=float)
        channels = grouped.iloc[:, 1:].to_numpy(dtype=float)
        print(f"Warning: averaged duplicate ECS depths; {len(ecs_depth)} rows remain")

    inside = (xrd_depth >= ecs_depth[0]) & (xrd_depth <= ecs_depth[-1])
    if not inside.all():
        if out_of_range == "error":
            raise ValueError(
                f"{int((~inside).sum())} XRD depths are outside the ECS range "
                f"[{ecs_depth[0]}, {ecs_depth[-1]}]. Extrapolation is disabled."
            )
        print(f"Warning: dropped {int((~inside).sum())} XRD rows outside the ECS depth range")
        xrd_depth = xrd_depth[inside]
        minerals = minerals.loc[inside].reset_index(drop=True)

    if len(ecs_depth) < 2:
        raise ValueError("At least two unique ECS depths are required")
    if len(xrd_depth) == 0:
        raise ValueError("No XRD samples remain within the ECS depth range")

    interpolator = interp1d(
        ecs_depth,
        channels,
        kind="linear",
        axis=0,
        bounds_error=True,
        assume_sorted=True,
    )
    aligned_channels = np.asarray(interpolator(xrd_depth), dtype=float)
    channel_frame = pd.DataFrame(
        aligned_channels,
        columns=INELASTIC_COLUMNS + CAPTURE_COLUMNS,
    )
    result = pd.concat(
        [pd.Series(xrd_depth, name="Depth"), channel_frame, minerals.reset_index(drop=True)],
        axis=1,
    )
    if well_id is not None:
        result.insert(0, "Well_ID", well_id)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False)
    print(
        f"Saved {output_path}: {len(result)} aligned samples, "
        "514 inelastic channels, 514 capture channels, and 8 mineral labels"
    )
    return result
