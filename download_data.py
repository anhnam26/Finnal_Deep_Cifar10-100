"""Download/verify official CIFAR archives only. No model, CUDA or training code."""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.utils import check_integrity, extract_archive


ROOT = Path(__file__).resolve().parent
DATASETS = {"cifar10": (CIFAR10, 10), "cifar100": (CIFAR100, 100)}


def md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(path, state):
    value = {**state, "updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def transfer(url, partial, report, timeout=60):
    """Resume only with validated Content-Range. Never append a full response."""
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "CIFAR-final-download/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset:
                raise ValueError("Server returned invalid Content-Range; refusing to append")
            total = int(match[3])
            if int(match[2]) < offset or int(match[2]) >= total:
                raise ValueError("Invalid range end/total")
            mode = "ab"
            resumed = offset > 0
        elif response.status == 200:
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
            if offset:
                print("Server ignored HTTP Range; restarting this archive from byte 0.", flush=True)
            mode, offset, resumed = "wb", 0, False
        else:
            raise ValueError(f"Unexpected HTTP status {response.status}")
        downloaded = offset
        start, last_report = time.monotonic(), 0.0
        with partial.open(mode) as file:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    file.flush()
                    report({"phase": "downloading", "bytes": downloaded, "total_bytes": total,
                            "resumed": resumed, "bytes_per_second": (downloaded - offset) / max(now - start, 0.001)})
                    last_report = now
        if total is not None and downloaded != total:
            raise IOError(f"Incomplete response: {downloaded}/{total} bytes")
    return downloaded


def fetch_archive(url, archive, expected_md5, report, attempts=5):
    if archive.exists() and md5(archive) == expected_md5:
        report({"phase": "archive_verified", "bytes": archive.stat().st_size})
        return
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.exists():
        if partial.exists():
            raise FileExistsError(f"Both incomplete archive and .part exist; inspect before proceeding: {archive}")
        archive.rename(partial)
    for attempt in range(1, attempts + 1):
        try:
            report({"phase": "connecting", "attempt": attempt, "bytes": partial.stat().st_size if partial.exists() else 0})
            transfer(url, partial, report)
            report({"phase": "checking_md5", "bytes": partial.stat().st_size})
            actual = md5(partial)
            if actual != expected_md5:
                # Retain bad data for diagnosis rather than publishing a bad archive.
                quarantine = partial.with_name(partial.name + f".bad-{time.time_ns()}")
                partial.rename(quarantine)
                raise ValueError(f"Official MD5 mismatch: {actual}; retained at {quarantine}")
            os.replace(partial, archive)
            report({"phase": "archive_verified", "bytes": archive.stat().st_size, "md5": actual})
            return
        except (OSError, ValueError) as error:
            report({"phase": "retry_wait" if attempt < attempts else "failed",
                    "attempt": attempt, "error": str(error),
                    "bytes": partial.stat().st_size if partial.exists() else 0})
            if attempt == attempts:
                raise
            time.sleep(min(30, 5 * attempt))


def verify_extracted(cls, classes, root):
    """Check official checksums and structure, not model performance."""
    details = {}
    for training, expected in ((True, 50000), (False, 10000)):
        dataset = cls(root=str(root), train=training, download=False)
        if len(dataset) != expected or dataset.data.shape != (expected, 32, 32, 3):
            raise ValueError("Unexpected official CIFAR image counts/shape")
        if set(dataset.targets) != set(range(classes)):
            raise ValueError("Unexpected label range")
        counts = [dataset.targets.count(label) for label in range(classes)]
        if any(count != expected // classes for count in counts):
            raise ValueError("Unexpected per-class counts")
        details["train" if training else "test"] = {"images": expected, "shape": list(dataset.data.shape), "classes": classes}
        del dataset
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--root", default=str(ROOT / "datasets"))
    parser.add_argument("--status-dir", default=str(ROOT / "outputs" / "downloads"))
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("attempts must be positive")
    root, status_dir = Path(args.root).resolve(), Path(args.status_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{args.dataset}.json"
    lock = root / f".{args.dataset}.download.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"Download lock exists: {lock}; confirm its PID stopped before removing a stale lock")
    os.close(fd)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    cls, classes = DATASETS[args.dataset]
    url = cls.url.replace("http://", "https://", 1)
    state = {"dataset": args.dataset, "source_url": url, "expected_md5": cls.tgz_md5,
             "data_root": str(root), "training_started": False}

    def report(update):
        nonlocal state
        state = {**state, **update}
        write_status(status_path, state)
        print(json.dumps(update), flush=True)

    try:
        archive = root / cls.filename
        fetch_archive(url, archive, cls.tgz_md5, report, args.attempts)
        # Avoid overwriting already verified extracted data on repeated runs.
        files = cls.train_list + cls.test_list + [(cls.meta["filename"], cls.meta["md5"])]
        if not all(check_integrity(str(root / cls.base_folder / name), checksum) for name, checksum in files):
            report({"phase": "extracting"})
            extract_archive(str(archive), str(root))
        report({"phase": "verifying_extracted_files"})
        details = verify_extracted(cls, classes, root)
        report({"phase": "completed", "verified": True, "structure": details, "error": None})
    except Exception as error:
        report({"phase": "failed", "verified": False, "error": str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()