# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Merge the three public CCTV accident datasets into one train/val/test set.

The paper combines:
  1. Accident Detection from CCTV Footage (Kaggle, Charan Kumar C.)   462 accident / 528 non-accident
  2. CCTV Accident Data Korea (Kaggle, Chang Hoon Park)            12,904 / 22,394
  3. Road Accidents from CCTV Footages (Kaggle, Ragavendiran et al.)  6,191 / 15,420
into Merged_Accident_Dataset: 27,368 train, 7,054 val, 7,054 test images.

Each source stores images under folders whose names say accident / non-accident,
sometimes inside train/val/test sub-folders. We collect every image, de-duplicate by
content hash, then make one stratified split so no image appears in two splits.

    python -m tcd.data merge            # writes data/Merged_Accident_Dataset
    python -m tcd.data stats            # prints counts and class weights
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

from . import CLASSES, ROOT, load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def label_from_path(p: Path) -> str | None:
    """Return 'Accident' / 'NonAccident' from any folder name on the path, else None."""
    for part in reversed([x.lower().replace("-", "").replace("_", "").replace(" ", "") for x in p.parts[:-1]]):
        if part in {"nonaccident", "noaccident", "normal", "nonaccidents"}:
            return "NonAccident"
        if part in {"accident", "accidents", "crash"}:
            return "Accident"
    return None


def collect(sources: list[str | Path]) -> list[tuple[Path, str]]:
    seen, items = set(), []
    for src in sources:
        for p in sorted(Path(src).rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab = label_from_path(p)
            if lab is None:
                continue
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen:                      # exact duplicates across sources
                continue
            seen.add(h)
            items.append((p, lab))
    return items


def stratified_split(items, split: dict, seed: int = 42):
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for lab in CLASSES:
        group = [it for it in items if it[1] == lab]
        rng.shuffle(group)
        n = len(group)
        n_val = round(n * split["val"])
        n_test = round(n * split["test"])
        out["val"] += group[:n_val]
        out["test"] += group[n_val:n_val + n_test]
        out["train"] += group[n_val + n_test:]
    return out


def class_weights(labels: list[str]) -> dict[int, float]:
    """Balanced weights w_c = N / (K * n_c), the form Keras expects.

    With the paper's training counts this gives 1.5849 for accidents and 0.7304 for
    non-accidents, matching Section 4.1.
    """
    c = Counter(labels)
    n, k = len(labels), len(CLASSES)
    return {i: n / (k * c[lab]) for i, lab in enumerate(CLASSES) if c[lab]}


def merge(cfg: dict) -> dict:
    out_dir = ROOT / cfg["data"]["merged_dir"]
    items = collect([ROOT / s for s in cfg["data"]["sources"]])
    splits = stratified_split(items, cfg["data"]["split"], cfg["data"]["seed"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, rows in splits.items():
        for i, (p, lab) in enumerate(rows):
            dst = out_dir / split / lab / f"{i:06d}{p.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    return {s: Counter(l for _, l in r) for s, r in splits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "stats"])
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "merge":
        print(merge(cfg))
    else:
        base = ROOT / cfg["data"]["merged_dir"]
        for split in ("train", "val", "test"):
            print(split, {lab: len(list((base / split / lab).glob("*"))) for lab in CLASSES})
        labels = [lab for lab in CLASSES for _ in (base / "train" / lab).glob("*")]
        print("class weights", class_weights(labels))


if __name__ == "__main__":
    main()
