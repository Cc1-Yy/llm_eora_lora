from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_ROOT = PROJECT_ROOT / "configs" / "cls"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def ensure_data_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = cfg.get("data")
    if not isinstance(data, dict):
        data = {}
        cfg["data"] = data
    return data


def process_file(path: Path, split_seed: int, force: bool, make_backup: bool) -> str:
    cfg = load_yaml(path)
    data = ensure_data_dict(cfg)

    if "split_seed" in data and not force:
        return f"skip   {path}  (already has data.split_seed={data['split_seed']})"

    old_value = data.get("split_seed", None)
    data["split_seed"] = int(split_seed)

    if make_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    save_yaml(cfg, path)

    if old_value is None:
        return f"update {path}  (added data.split_seed={split_seed})"
    return f"update {path}  (changed data.split_seed: {old_value} -> {split_seed})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_CFG_ROOT),
        help="Root directory to search for yaml config files.",
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Value to set for data.split_seed.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data.split_seed.",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Config root not found: {root}")

    yaml_files = sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml")))
    if not yaml_files:
        print(f"No yaml files found under: {root}")
        return

    print(f"Found {len(yaml_files)} yaml files under: {root}")
    print(f"Setting data.split_seed = {args.split_seed}")
    print()

    updated = 0
    skipped = 0

    for path in yaml_files:
        msg = process_file(
            path=path,
            split_seed=args.split_seed,
            force=args.force,
            make_backup=not args.no_backup,
        )
        print(msg)
        if msg.startswith("update"):
            updated += 1
        else:
            skipped += 1

    print()
    print(f"Done. updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()