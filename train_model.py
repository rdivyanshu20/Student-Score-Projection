from __future__ import annotations

import csv
from pathlib import Path


def load_dataset(file_path: Path) -> list[tuple[float, float]]:
    """Load study hours and exam scores from a CSV file."""
    rows: list[tuple[float, float]] = []

    with file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            hours = float(row["hours_studied"])
            score = float(row["exam_score"])
            rows.append((hours, score))

    return rows


def split_dataset(rows: list[tuple[float, float]], train_ratio: float = 0.8) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split the dataset into training and testing parts."""
    split_index = int(len(rows) * train_ratio)
    training_rows = rows[:split_index]
    testing_rows = rows[split_index:]
    return training_rows, testing_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def fit_linear_regression(rows: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Fit a simple linear regression model:
    y = slope * x + intercept
    """
    x_values = [row[0] for row in rows]
    y_values = [row[1] for row in rows]

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in rows)
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict(hours_studied: float, slope: float, intercept: float) -> float:
    return slope * hours_studied + intercept


def mean_absolute_error(rows: list[tuple[float, float]], slope: float, intercept: float) -> float:
    errors = [abs(actual - predict(hours, slope, intercept)) for hours, actual in rows]
    return mean(errors)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "student_scores.csv"

    rows = load_dataset(dataset_path)
    training_rows, testing_rows = split_dataset(rows, train_ratio=0.8)

    slope, intercept = fit_linear_regression(training_rows)
    test_mae = mean_absolute_error(testing_rows, slope, intercept)

    print("Student Score Prediction")
    print("-" * 30)
    print(f"Total rows: {len(rows)}")
    print(f"Training rows: {len(training_rows)}")
    print(f"Testing rows: {len(testing_rows)}")
    print()
    print("Learned model:")
    print(f"predicted_score = {slope:.2f} * hours_studied + {intercept:.2f}")
    print()
    print(f"Test mean absolute error: {test_mae:.2f}")
    print()
    print("Predictions on the test set:")
    for hours, actual_score in testing_rows:
        predicted_score = predict(hours, slope, intercept)
        print(
            f"Hours studied: {hours:>4.1f} | "
            f"Actual score: {actual_score:>5.1f} | "
            f"Predicted score: {predicted_score:>6.2f}"
        )

    print()
    my_hours = 6.0
    my_prediction = predict(my_hours, slope, intercept)
    print(f"If a student studies {my_hours:.1f} hours, the model predicts a score of {my_prediction:.2f}.")


if __name__ == "__main__":
    main()
