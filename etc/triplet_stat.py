import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("human_eval_triplet.csv")

    x = list(df["rel1"]) + list(df["rel2"])
    x = np.array(x)


    print(x.mean())
    print(x.std())


if __name__ == "__main__":
    main()