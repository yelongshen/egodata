from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .catalog import Dataset


CHUNK_SIZE = 1024 * 1024


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _archive_id(path: Path) -> str:
    name = path.name
    for suffix in (".tar.bz2", ".tar.gz", ".tar.xz", ".tar", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def dataset_local_paths(dataset: Dataset, root: Path) -> list[Path]:
    dataset_dir = root / dataset.slug
    filenames = dataset.tracked_files or tuple(asset.filename for asset in dataset.assets)
    return [dataset_dir / filename for filename in filenames]


def dataset_local_status(dataset: Dataset, root: Path) -> tuple[list[Path], list[Path]]:
    expected_paths = dataset_local_paths(dataset, root)
    present = [path for path in expected_paths if path.exists()]
    missing = [path for path in expected_paths if not path.exists()]
    return present, missing


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    resolved_destination = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if resolved_destination not in member_path.parents and member_path != resolved_destination:
                raise ValueError(f"Unsafe archive member path in {archive_path}: {member.name}")
        archive.extractall(destination)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.name.endswith((".tar.bz2", ".tar.gz", ".tar.xz", ".tar")):
        _safe_extract_tar(archive_path, destination)
        return
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def extract_local_dataset(
    dataset: Dataset,
    root: Path,
    destination_root: Path | None = None,
    archive_names: set[str] | None = None,
    force: bool = False,
) -> list[Path]:
    dataset_dir = root / dataset.slug
    destination_root = destination_root or (dataset_dir / "extracted")
    ensure_dir(destination_root)

    extracted_paths: list[Path] = []
    for archive_path in dataset_local_paths(dataset, root):
        archive_name = _archive_id(archive_path)
        if archive_names is not None and archive_name not in archive_names and archive_path.name not in archive_names:
            continue
        if not archive_path.exists():
            continue

        target_dir = destination_root / archive_name
        if target_dir.exists() and any(target_dir.iterdir()) and not force:
            extracted_paths.append(target_dir)
            continue

        ensure_dir(target_dir)
        _extract_archive(archive_path, target_dir)
        extracted_paths.append(target_dir)

    return extracted_paths


def _remote_content_length(url: str) -> int | None:
    """Return the Content-Length of a URL via HEAD, or None if unavailable."""
    request = Request(url, headers={"User-Agent": "motion-dataset-downloaders/0.1"}, method="HEAD")
    try:
        with urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except HTTPError:
        return None


def download_url(url: str, destination: Path, force: bool = False) -> None:
    """Download url to destination, resuming partial files and retrying on transient errors."""
    import time

    remote_size = _remote_content_length(url)

    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        local_size = destination.stat().st_size if destination.exists() else 0

        # Already complete — skip.
        if not force and remote_size is not None and local_size >= remote_size:
            return

        try:
            if local_size > 0 and not force:
                # Attempt a Range resume from where we left off.
                headers = {
                    "User-Agent": "motion-dataset-downloaders/0.1",
                    "Range": f"bytes={local_size}-",
                }
                request = Request(url, headers=headers)
                with urlopen(request) as response:
                    if response.status == 206:
                        with destination.open("ab") as handle:
                            while True:
                                chunk = response.read(CHUNK_SIZE)
                                if not chunk:
                                    break
                                handle.write(chunk)
                        return
                    # Server ignored Range (200) — fall through to full download.

            # Full download (new file, force, or server doesn't support Range).
            request = Request(url, headers={"User-Agent": "motion-dataset-downloaders/0.1"})
            with urlopen(request) as response, destination.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
            return

        except (OSError, ConnectionError) as exc:
            if attempt == max_attempts:
                raise
            wait = min(30 * attempt, 300)  # 30s, 60s, 90s … capped at 5 min
            print(f"[attempt {attempt}/{max_attempts}] {exc!r} — retrying in {wait}s …",
                  flush=True)
            time.sleep(wait)


def download_dataset(
    dataset: Dataset,
    root: Path,
    force: bool = False,
    asset_names: set[str] | None = None,
) -> list[Path]:
    if dataset.access != "public_direct":
        raise ValueError(
            f"Dataset '{dataset.slug}' is not directly downloadable. Access type: {dataset.access}"
        )

    dataset_dir = root / dataset.slug
    ensure_dir(dataset_dir)

    written_paths: list[Path] = []
    for asset in dataset.assets:
        if asset_names is not None and asset.name not in asset_names:
            continue
        destination = dataset_dir / asset.filename
        download_url(asset.url, destination, force=force)
        written_paths.append(destination)

    return written_paths