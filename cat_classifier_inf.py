"""Run inference: predict which cat is in a given image."""

from pathlib import Path

import typer

from cat_classifier import CatClassifier, extract_features


def main(
    image: Path = typer.Option(..., help="Path to the input image."),
    model: Path = typer.Option(Path("models/cat_classifier.joblib"), help="Path to a trained model file."),
) -> None:
    """Predict which cat appears in IMAGE using a trained MODEL."""
    clf = CatClassifier.load(model)

    features = extract_features(image).reshape(1, -1)
    predicted_class = clf.predict(features)[0]
    probas = clf.predict_proba(features)[0]
    confidence = probas[clf.classes_.index(predicted_class)]

    typer.echo(f"Predicted: {predicted_class} ({confidence:.0%} confident)")


if __name__ == "__main__":
    typer.run(main)
