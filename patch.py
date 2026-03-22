"""
Patch local MNE CNT readers to avoid int32 overflow in size arithmetic.

Usage:
    python patch.py
"""

from pathlib import Path
import sysconfig

LOCAL_MNE_CNT_REL = Path("mne/io/cnt/cnt.py")

REPLACEMENTS = [
    (
        "data_size = n_samples * n_channels",
        "data_size = np.int64(n_samples) * np.int64(n_channels)",
    ),
    (
        "data_size = event_offset - (data_offset + 75 * n_channels)",
        "data_size = np.int64(event_offset) - (np.int64(data_offset) + 75 * np.int64(n_channels))",
    ),
    (
        "if n_samples == 0 or data_size // (n_samples * n_channels) not in [2, 4]:",
        "if np.int64(n_samples) == 0 or np.int64(data_size) // (np.int64(n_samples) * np.int64(n_channels)) not in [2, 4]:",
    ),
    (
        "n_samples = data_size // (n_bytes * n_channels)",
        "n_samples = np.int64(data_size) // (np.int64(n_bytes) * np.int64(n_channels))",
    ),
    (
        "n_bytes = data_size // (n_samples * n_channels)",
        "n_bytes = np.int64(data_size) // (np.int64(n_samples) * np.int64(n_channels))",
    ),
]


def patch_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"SKIP {path} (missing)"

    original = path.read_text(encoding="utf-8")
    updated = original
    applied = 0

    for old, new in REPLACEMENTS:
        if old in updated:
            updated = updated.replace(old, new)
            applied += 1

    if updated == original:
        if any(new in original for _, new in REPLACEMENTS):
            return False, f"OK   {path} (already patched)"
        return False, f"WARN {path} (no expected patterns found)"

    path.write_text(updated, encoding="utf-8")
    return True, f"DONE {path} ({applied} replacements)"


def discover_target_files() -> list[Path]:
    """Find potential MNE cnt.py files across Linux and Windows venv layouts."""
    candidates: list[Path] = []

    # Python currently executing this script.
    for key in ("purelib", "platlib"):
        site_dir = sysconfig.get_paths().get(key)
        if site_dir:
            candidates.append(Path(site_dir) / LOCAL_MNE_CNT_REL)

    # Common project-local virtualenv layouts.
    venv_root = Path(".venv")
    candidates.append(venv_root / "Lib" / "site-packages" / LOCAL_MNE_CNT_REL)  # Windows

    for pattern in (
        ".venv/lib/python*/site-packages/mne/io/cnt/cnt.py",
        ".venv/lib64/python*/site-packages/mne/io/cnt/cnt.py",
    ):
        candidates.extend(Path().glob(pattern))

    # Deduplicate while preserving discovery order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        normalized = path.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def main() -> int:
    target_files = discover_target_files()
    if not target_files:
        print("No candidate MNE cnt.py files were discovered.")
        return 1

    changed = 0
    for file_path in target_files:
        was_changed, message = patch_file(file_path)
        print(message)
        if was_changed:
            changed += 1

    if changed == 0:
        print("No files changed.")
    else:
        print(f"Patched {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
