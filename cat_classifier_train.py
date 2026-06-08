"""Train the cat classifier and save the model to disk."""

from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import accuracy_score

from cat_classifier import CatClassifier, extract_features


def collect_images(data_dir: Path, skip: tuple[str, ...] = ()) -> tuple[np.ndarray, list[str]]:
    """Walk each subdirectory of data_dir; use the folder name as the class label."""
    X_rows: list[np.ndarray] = []
    y_labels: list[str] = []

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        if label in skip:
            typer.echo(f"  skipping {class_dir} (excluded)")
            continue
        images = sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.jpeg"))
        if not images:
            typer.echo(f"  warning: {class_dir} has no .jpeg files — skipping")
            continue
        for img_path in images:
            X_rows.append(extract_features(img_path))
            y_labels.append(label)

    if not X_rows:
        raise typer.Exit(typer.echo("Error: no images found. Add .jpeg files under subdirectories of data/") or 1)

    return np.stack(X_rows), y_labels


def main(
    data_dir: Path = typer.Option(Path("data/"), help="Root data directory; each subdirectory is treated as one class."),
    model_out: Path = typer.Option(Path("models/cat_classifier.joblib"), help="Where to save the trained model."),
    skip: list[str] = typer.Option(["test"], help="Subdirectory names to ignore (repeatable)."),
) -> None:
    """Train a cat classifier on images in DATA_DIR and save to MODEL_OUT."""
    typer.echo(f"Loading images from {data_dir} ...")
    X, y = collect_images(data_dir, skip=tuple(skip))
    counts = {label: y.count(label) for label in sorted(set(y))}
    typer.echo(f"  {len(y)} images found: {counts}")

    clf = CatClassifier()
    clf.fit(X, y)

    acc = accuracy_score(y, clf.predict(X))
    typer.echo(f"Train accuracy: {acc:.1%}")

    clf.save(model_out)
    typer.echo(f"Model saved to {model_out}")


if __name__ == "__main__":
    typer.run(main)
