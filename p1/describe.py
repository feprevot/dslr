import os 
import pandas as pd


def describe(DATASET):
    """Describe the dataset."""

    df = pd.read_csv(DATASET)
    print(df.head())

def main():
    if len(os.sys.argv) < 2:
        print("Usage: python describe.py <dataset_path>")
        return
    dataset = os.sys.argv[1]
    describe(dataset)

if __name__ == "__main__":
    main()