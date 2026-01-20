import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === CONFIG ===
PARQUET_PATH = "recipe_embeddings.parquet"
FAISS_INDEX_PATH = "recipe_faiss.index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5

# load embedding DB
print("Loading dataset & FAISS...")
df = pd.read_parquet(PARQUET_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

def recommend_recipe(query: str, k: int = TOP_K):
    # embed query
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    # search
    distances, indices = index.search(query_emb, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        row = df.iloc[idx]
        results.append({
            "Title": row["Title"],
            "Ingredients": row["Ingredients"],
            "Steps": row["Steps"],
            "Distance": float(dist)
        })
    return results

if __name__ == "__main__":
    print("Chatbot recipe ready.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit","quit","bye"]:
            print("bye!")
            break
        
        recs = recommend_recipe(user_input)
        print("\nTop Recommendations:\n")
        for r in recs:
            print(f"Title: {r['Title']}")
            print(f"Ingredients: {r['Ingredients'][:200]} ...")
            print(f"Steps: {r['Steps'][:200]} ...")
            print("-"*80)
        print()
