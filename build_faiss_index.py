import pandas as pd
import numpy as np
import faiss

PARQUET_PATH = "recipe_embeddings.parquet"
OUTPUT_INDEX = "recipe_faiss.index"

df = pd.read_parquet(PARQUET_PATH)

embeddings = np.array(df["embedding"].tolist()).astype("float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, OUTPUT_INDEX)

print("DONE!")
print("FAISS index saved:", OUTPUT_INDEX)
