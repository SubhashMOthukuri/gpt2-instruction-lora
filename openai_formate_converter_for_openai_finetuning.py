import pandas as pd

# Load the dataset
df = pd.read_csv("bbc_news.csv")  # change filename if needed
df = df.dropna(subset=["article", "summary"])  # remove rows with missing data


import json

with open("bbc_finetune.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        article = row["article"]
        summary = row["summary"]

        json_obj = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
                {"role": "user", "content": f"Summarize this article:\n\n{article.strip()}"},
                {"role": "assistant", "content": summary.strip()}
            ]
        }

        f.write(json.dumps(json_obj) + "\n")
