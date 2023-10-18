import pandas as pd


def main():
    x = [{"a": 1, "b": 2, "c": 3}, {"a": 2, "b": 4, "c": 6}]
    df = pd.DataFrame(x)

    print(df)

if __name__ == "__main__":
    main()