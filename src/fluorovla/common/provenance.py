"""Provenance helpers: SHA-256 file manifests, environment capture and the run manifest every run directory must contain."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths) -> dict[str, str]:
    return {str(p).replace("\\", "/"): (sha256_file(p) if Path(p).is_file() else "MISSING") for p in paths}


def write_file_manifest(root: str | Path, manifest: str | Path) -> int:
    """Write '<sha256>  <relative path>' lines for every file under ``root`` (except the manifest itself). Returns the number of files."""
    root, manifest = Path(root), Path(manifest)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() != manifest.resolve())
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n")
    return len(files)


def verify_file_manifest(root: str | Path, manifest: str | Path) -> dict:
    """Verify a manifest; returns {'checked', 'bad' (mismatching or missing), 'extra'}."""
    root, manifest = Path(root), Path(manifest)
    want = {}
    for line in open(manifest, encoding="utf-8"):
        if line.strip():
            h, r = line.rstrip("\n").split("  ", 1)
            want[r] = h
    bad = [r for r, h in want.items() if not (root / r).is_file() or sha256_file(root / r) != h]
    have = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.resolve() != manifest.resolve()}
    return {"checked": len(want), "bad": bad, "extra": sorted(have - set(want))}


def capture_environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "kernel": platform.release(), "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for mod in ("warp", "numpy", "scipy", "skimage", "newton"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception as ex:  # noqa: BLE001
            env[mod] = f"unavailable ({type(ex).__name__})"
    try:
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        env["gpu"] = "unavailable"
    return env


def code_hash(package_dir: str | Path | None = None) -> dict:
    """SHA-256 of every source file of the package plus one combined hash."""
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parents[1]
    files = sorted(p for p in root.rglob("*.py"))
    per = {p.relative_to(root).as_posix(): sha256_file(p) for p in files}
    combined = hashlib.sha256("".join(f"{k}{v}" for k, v in per.items()).encode()).hexdigest()
    return {"combined": combined, "files": per}


def write_run_manifest(run_dir: str | Path, purpose: str, patients: list[str], config: dict, status: str, extra: dict | None = None) -> Path:
    """Write ``MANIFEST.json`` (purpose, patients, config, code hash, environment, timestamp, status). Refuses to overwrite: use a new run directory or attempt."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    f = run_dir / "MANIFEST.json"
    if f.exists():
        raise FileExistsError(f"{f} exists; evidence is never overwritten")
    rec = {"purpose": purpose, "patients": patients, "config": config, "code_sha256": code_hash()["combined"], "environment": capture_environment(),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "cwd": os.getcwd()}
    rec.update(extra or {})
    json.dump(rec, open(f, "w"), indent=1, default=str)
    return f
