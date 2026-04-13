import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/dataset_train.csv")
HOUSE_COLORS = {
    "Gryffindor": "#c9362c",
    "Hufflepuff": "#f0c532",
    "Ravenclaw": "#2660a4",
    "Slytherin": "#1a7f44",
}

def main():
    df = pd.read_csv(DATASET_PATH)

    feat_x = "Astronomy"
    feat_y = "Defense Against the Dark Arts"
    
    _, ax = plt.subplots(figsize=(8, 6))

    for house, color in HOUSE_COLORS.items():
        subset = df[df["Hogwarts House"] == house]
        ax.scatter(
            subset[feat_x],
            subset[feat_y],
            label=house,
            color=color,
            alpha=0.6,
            s=10,
        )

    ax.set_title(
        f"Most similar features: {feat_x} vs {feat_y}\n"
    )
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.legend(title="House")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
