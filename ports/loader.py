# ports/loader.py
import pandas as pd

def load_reviews_from_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    return df["review"].tolist()
