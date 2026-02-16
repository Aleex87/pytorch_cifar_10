import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def unpickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_class_names(meta_path: Path) -> list[str]:
    meta = unpickle(meta_path)
    # In CIFAR-10, class names are stored under b"label_names"
    return [name.decode("utf-8") for name in meta[b"label_names"]]


def save_split(
    batch_files: list[Path],
    out_root: Path,
    class_names: list[str],
    prefix: str,
):
    """
    Reads CIFAR-10 batch files and saves images into:
      pytorch/data/processed/<index_test>,<index_train>/.png
    """
    out_root.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    for bf in batch_files:
        batch = unpickle(bf)

        data = batch[b"data"]          # shape (N, 3072)
        labels = batch[b"labels"]      # length N

        # Convert to (N, 32, 32, 3)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        for img_arr, y in zip(data, labels):
            cls = class_names[y]
            cls_dir = out_root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)

            img = Image.fromarray(img_arr.astype(np.uint8))
            img_path = cls_dir / f"{prefix}_{global_idx:06d}.png"
            img.save(img_path)

            global_idx += 1


def main():
    raw_dir = Path("data/raw/cifar-10-batches-py")
    meta_path = raw_dir / "batches.meta"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Non trovo {raw_dir}. Sei nella root del progetto?")

    class_names = load_class_names(meta_path)

    # Train: data_batch_1..5
    train_batches = [raw_dir / f"data_batch_{i}" for i in range(1, 6)]
    # Test: test_batch
    test_batches = [raw_dir / "test_batch"]

    out_train = Path("data/processed/train")
    out_test = Path("data/processed/test")

    print("Salvo TRAIN in:", out_train)
    save_split(train_batches, out_train, class_names, prefix="train")

    print("Salvo TEST in:", out_test)
    save_split(test_batches, out_test, class_names, prefix="test")

    print("Fatto.")


if __name__ == "__main__":
    main()
