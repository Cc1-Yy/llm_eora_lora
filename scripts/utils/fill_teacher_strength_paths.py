import re
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "sweep_teacher_strength_lm"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EORA_YAML_RE = re.compile(r"^eora_vs_teacher_steps(\d+)\.yaml$", re.IGNORECASE)

def find_teacher_model_dir(steps: int) -> Path:
    root = OUTPUTS_DIR / f"teacher_steps{steps}" / "runs"
    if not root.exists():
        raise FileNotFoundError(f"Teacher runs folder not found: {root}")

    # pick latest modified run folder
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs under: {root}")

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run = run_dirs[0]

    # prefer model_best if exists else model
    mb = run / "model_best"
    m = run / "model"
    if mb.exists():
        return mb
    if m.exists():
        return m
    raise FileNotFoundError(f"No model_best/ or model/ under run: {run}")

def main():
    updated = 0
    for p in CONFIG_DIR.iterdir():
        if not p.is_file():
            continue
        m = EORA_YAML_RE.match(p.name)
        if not m:
            continue
        steps = int(m.group(1))

        teacher_model_dir = find_teacher_model_dir(steps)
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["optimized_model_dir"] = str(teacher_model_dir).replace("\\", "/")

        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(f"[OK] {p.name}: optimized_model_dir -> {cfg['optimized_model_dir']}")
        updated += 1

    print(f"Updated {updated} YAMLs.")

if __name__ == "__main__":
    main()