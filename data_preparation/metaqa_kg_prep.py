import pandas as pd
import json

kg_data = pd.read_csv("data/MetaQA/MetaQA/kb.txt", sep="|", names=["subject", "predicate", "object"])
kg_data.to_parquet("data/MetaQA/MetaQA/kb.parquet")
