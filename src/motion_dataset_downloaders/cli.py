from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog import ALL_DATASETS, DATASETS, get_dataset, iter_by_access
from .downloader import dataset_local_status, download_dataset, extract_local_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion-datasets",
        description="Dataset planner/downloader for public motion and hand datasets related to EgoScale and SONIC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List all tracked datasets")
    list_parser.add_argument(
        "--access",
        choices=["public_direct", "manual_license", "external_public", "unavailable"],
        help="Filter by access type.",
    )

    plan_parser = subparsers.add_parser("plan", help="Summarize datasets by access type")
    plan_parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Include non-public datasets in the report.",
    )

    info_parser = subparsers.add_parser("info", help="Show detailed information about one dataset")
    info_parser.add_argument("slug", help="Dataset slug")

    local_status_parser = subparsers.add_parser(
        "local-status", help="Show which tracked local files are present for a dataset"
    )
    local_status_parser.add_argument("slug", help="Dataset slug")
    local_status_parser.add_argument("--root", default="downloads", help="Download directory root")

    download_parser = subparsers.add_parser("download", help="Download one directly downloadable dataset")
    download_parser.add_argument("slug", help="Dataset slug")
    download_parser.add_argument("--root", default="downloads", help="Download directory root")
    download_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    download_parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Only download the named asset. Can be repeated.",
    )

    download_public_parser = subparsers.add_parser(
        "download-public", help="Download all datasets with direct public URLs"
    )
    download_public_parser.add_argument("--root", default="downloads", help="Download directory root")
    download_public_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    download_public_parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Restrict downloads to the named asset for datasets that provide it.",
    )

    extract_parser = subparsers.add_parser(
        "extract-local", help="Extract already downloaded local archives for a dataset"
    )
    extract_parser.add_argument("slug", help="Dataset slug")
    extract_parser.add_argument("--root", default="downloads", help="Download directory root")
    extract_parser.add_argument(
        "--dest",
        help="Destination directory for extracted files. Defaults to <root>/<slug>/extracted",
    )
    extract_parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Only extract the named archive or archive stem. Can be repeated.",
    )
    extract_parser.add_argument("--force", action="store_true", help="Re-extract into existing directories")

    return parser


def _print_dataset_line(dataset) -> None:
    print(f"{dataset.slug:24} {dataset.access:16} {dataset.name}")


def cmd_list(args: argparse.Namespace) -> int:
    datasets = ALL_DATASETS
    if args.access:
        datasets = iter_by_access(args.access)
    for dataset in datasets:
        _print_dataset_line(dataset)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    order = ["public_direct", "manual_license", "external_public"]
    if args.include_unavailable:
        order.append("unavailable")

    for access in order:
        print(f"\n[{access}]")
        datasets = iter_by_access(access)
        if not datasets:
            print("  (none)")
            continue
        for dataset in datasets:
            print(f"- {dataset.slug}: {dataset.name}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.slug)
    print(f"name: {dataset.name}")
    print(f"slug: {dataset.slug}")
    print(f"modality: {dataset.modality}")
    print(f"relation: {dataset.relation}")
    print(f"access: {dataset.access}")
    print(f"homepage: {dataset.homepage}")

    if dataset.assets:
        print("assets:")
        for asset in dataset.assets:
            print(f"- {asset.name}: {asset.url}")

    if dataset.notes:
        print("notes:")
        for note in dataset.notes:
            print(f"- {note}")

    if dataset.manual_steps:
        print("manual_steps:")
        for step in dataset.manual_steps:
            print(f"- {step}")
    if dataset.tracked_files:
        print("tracked_files:")
        for filename in dataset.tracked_files:
            print(f"- {filename}")
    return 0


def cmd_local_status(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.slug)
    present, missing = dataset_local_status(dataset, Path(args.root))

    print(f"dataset: {dataset.slug}")
    print(f"root: {Path(args.root) / dataset.slug}")
    print(f"present: {len(present)}")
    for path in present:
        print(f"- {path}")

    if missing:
        print(f"missing: {len(missing)}")
        for path in missing:
            print(f"- {path.name}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.slug)
    if dataset.access != "public_direct":
        print(
            f"Cannot auto-download '{dataset.slug}'. Access type is '{dataset.access}'.",
            file=sys.stderr,
        )
        if dataset.manual_steps:
            print("Manual steps:", file=sys.stderr)
            for step in dataset.manual_steps:
                print(f"- {step}", file=sys.stderr)
        return 2

    asset_names = set(args.asset) if args.asset else None
    paths = download_dataset(dataset, Path(args.root), force=args.force, asset_names=asset_names)
    if not paths:
        print("No assets matched the requested filter.", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


def cmd_download_public(args: argparse.Namespace) -> int:
    root = Path(args.root)
    asset_names = set(args.asset) if args.asset else None
    for dataset in DATASETS:
        if dataset.access != "public_direct":
            continue
        print(f"Downloading {dataset.slug} ...")
        paths = download_dataset(dataset, root, force=args.force, asset_names=asset_names)
        for path in paths:
            print(f"- {path}")
    return 0


def cmd_extract_local(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.slug)
    root = Path(args.root)
    destination = Path(args.dest) if args.dest else None
    archive_names = set(args.archive) if args.archive else None

    paths = extract_local_dataset(
        dataset,
        root,
        destination_root=destination,
        archive_names=archive_names,
        force=args.force,
    )
    if not paths:
        print("No local archives matched the request.", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return cmd_list(args)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "local-status":
        return cmd_local_status(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "download-public":
        return cmd_download_public(args)
    if args.command == "extract-local":
        return cmd_extract_local(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())