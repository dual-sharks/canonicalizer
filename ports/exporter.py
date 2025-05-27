# ports/exporter.py
import pandas as pd

def save_insights(insights: list[dict], path="output/insights.csv"):
    df = pd.DataFrame(insights)
    df.to_csv(path, index=False)
