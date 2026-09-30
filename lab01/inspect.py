"""
lab01/inspect.py
----------------
Modular data inspection utility for CSV datasets (local files or URLs).
Lab 01 Home Assignment.
"""
import sys
import pandas as pd


def inspect(path_or_url: str) -> pd.DataFrame:
    """Load a CSV and print a compact, standard first-look report.

    Parameters
    ----------
    path_or_url : str
        A local file path or an http(s) URL pointing at a CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.
    """
    df = pd.read_csv(path_or_url)

    sep = "=" * 70
    print(sep)
    print("SOURCE :", path_or_url)
    print("SHAPE  :", df.shape[0], "rows x", df.shape[1], "columns")
    print(sep)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("None.")
    else:
        report = pd.DataFrame({
            "missing": miss,
            "percent": (miss / len(df) * 100).round(2),
        })
        print(report.sort_values("missing", ascending=False))

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe())
    print(sep + "\n")

    return df


if __name__ == "__main__":
    datasets = [
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    ]
    for url in datasets:
        inspect(url)
