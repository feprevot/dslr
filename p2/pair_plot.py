import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/dataset_train.csv")
HOUSE_COLORS = {
    "Gryffindor": "#c9362c",
    "Hufflepuff": "#f0c532",
    "Ravenclaw": "#2660a4",
    "Slytherin": "#1a7f44",
}

FEATURES = [
    # "Arithmancy", # homogenous repartition between houses, don't keep it
    "Astronomy",
    "Herbology",
    # "Defense Against the Dark Arts", # same as Astronomy, don't keep it
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    # "Care of Magical Creatures", # homogenous repartition between houses, don't keep it
    "Charms",
    "Flying",
]


def main():
    df = pd.read_csv(DATASET_PATH)
    data = df[["Hogwarts House"] + FEATURES].dropna()
    sns.pairplot(data, hue="Hogwarts House", palette=HOUSE_COLORS, markers=".")
    plt.show()


if __name__ == "__main__":
    main()
