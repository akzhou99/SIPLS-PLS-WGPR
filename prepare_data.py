"""Generate net spectra and align them with XRD measurements."""

import argparse

from models.spectra import align, compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    net = actions.add_parser("net", help="Calculate net inelastic and capture spectra")
    net.add_argument("--input", required=True)
    net.add_argument("--output", required=True)
    net.add_argument("--well-id")
    net.add_argument("--missing-policy", choices=["error", "zero"], default="error")
    net.add_argument("--negative-policy", choices=["error", "clip"], default="clip")

    matching = actions.add_parser("align", help="Align net spectra to XRD depths")
    matching.add_argument("--ecs-file", required=True)
    matching.add_argument("--xrd-file", required=True)
    matching.add_argument("--output", required=True)
    matching.add_argument("--ecs-sheet", default="Sheet1")
    matching.add_argument("--xrd-sheet", default="XRD")
    matching.add_argument("--well-id")
    matching.add_argument("--out-of-range", choices=["error", "skip"], default="error")
    matching.add_argument("--duplicate-policy", choices=["error", "mean"], default="error")
    matching.add_argument("--ecs-no-header", action="store_true")
    matching.add_argument("--unlabelled-order", choices=["inelastic-first", "capture-first"])
    args = parser.parse_args()
    if args.action == "net":
        compute(args.input, args.output, well_id=args.well_id,
                missing_policy=args.missing_policy, negative_policy=args.negative_policy)
    else:
        align(ecs_file=args.ecs_file, xrd_file=args.xrd_file, output=args.output,
              ecs_sheet=args.ecs_sheet, xrd_sheet=args.xrd_sheet, well_id=args.well_id,
              out_of_range=args.out_of_range, duplicate_policy=args.duplicate_policy,
              ecs_no_header=args.ecs_no_header, unlabelled_order=args.unlabelled_order)


if __name__ == "__main__":
    main()
