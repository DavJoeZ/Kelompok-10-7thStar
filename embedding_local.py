import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

CSV_PATH = "dataset_recipes.csv"
OUTPUT_METADATA = "recipe_metadata.csv"
TEXT_COLUMN = "combined_text"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading dataset ...")
df = pd.read_csv(CSV_PATH)

if TEXT_COLUMN not in df.columns:
    print("Generating combined_text column ...")
    df[TEXT_COLUMN] = (
        df["Title"].astype(str) + " " +
        df["Ingredients"].astype(str) + " " +
        df["Steps"].astype(str)
    )

print("Loading model ...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding embeddings ...")
texts = df[TEXT_COLUMN].astype(str).tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

df["embedding"] = embeddings.tolist()

# save metadata CSV
# minimal 3 kolom ini sudah cukup untuk chatbot scr RAG
df[['Title','Ingredients','Steps','combined_text']].to_csv(OUTPUT_METADATA, index=False)

print("DONE!")
print("Saved metadata to:", OUTPUT_METADATA)
