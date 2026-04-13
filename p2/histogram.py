import os
import matplotlib.pyplot as plt
import pandas as pd

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = ["#c0392b", "#f39c12", "#2980b9", "#27ae60"]

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/dataset_train.csv")

def std(values):
    values = list(values)
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return (sum((x - m) ** 2 for x in values) / (n - 1)) ** 0.5


def homogeneity_score(df, course):
    """Lower is more homogeneous: mean of std across houses."""
    stds = []
    for house in HOUSES:
        scores = df[df["Hogwarts House"] == house][course].dropna()
        stds.append(std(scores))
    return sum(stds) / len(stds)

def main():
    df = pd.read_csv(DATASET_PATH)

    # Just keep course (exclude non-course columns)
    courses = [
        c for c in df.columns
        if c not in ("Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand")
    ]

    best_course = min(courses, key=lambda c: homogeneity_score(df, c))

    _, ax = plt.subplots(figsize=(8, 5))
    for house, color in zip(HOUSES, COLORS):
        scores = df[df["Hogwarts House"] == house][best_course].dropna()
        ax.hist(scores, bins=20, alpha=0.5, label=house, color=color)

    ax.set_title(f"Most homogeneous course: {best_course}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()