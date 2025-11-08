"""
app_final_ollama.py

Streamlit chat UI (bubble) + FAISS retrieval + MMR rerank + Ollama (deepseek-r1) integration.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import random
import faiss
from typing import List, Dict

# ====== CONFIG ======
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
FAISS_INDEX_PATH = "recipe_faiss.index"
METADATA_CSV = "recipe_metadata.csv"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # set sesuai embedding yang kamu pakai
TOP_K_RETRIEVE = 10   # initial candidates from FAISS
TOP_K_CONTEXT = 3     # contexts sent to model
TOP_K_RETURN = 5      # how many recipes to show when user asks details
MMR_LAMBDA = 0.6      # diversity vs relevance (0..1)
EMBED_DIM = 384       # change if your embedding dim differs
MODEL_OLLAMA = "deepseek-r1:1.5b"
MAX_CONTEXT_CHARS = 1400

# ====== UI PAGE ======
st.set_page_config(page_title="Chatbot Resep — Ollama", layout="wide")
st.title("🍳 Chatbot Recipe Recommendation — Ollama (deepseek-r1:1.5b)")

# ====== CACHED ASSETS LOADER ======
@st.cache_resource(show_spinner=False)
def load_assets():
    # load metadata CSV
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"{METADATA_CSV} not found in working dir.")
    df = pd.read_csv(METADATA_CSV)
    # load faiss index
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"{FAISS_INDEX_PATH} not found in working dir.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    # try load embeddings from parquet/np if present for exact MMR similarity computation
    embeddings = None
    # Try to find recipe_embeddings.parquet or embeddings.npy
    if os.path.exists("recipe_embeddings.parquet"):
        tmp = pd.read_parquet("recipe_embeddings.parquet")
        embeddings = np.array(tmp["embedding"].tolist()).astype("float32")
    elif os.path.exists("embeddings.npy"):
        embeddings = np.load("embeddings.npy").astype("float32")
    else:
        # if embeddings not present, build embeddings from metadata combined_text if sentence-transformers available
        embeddings = None
    return df, index, embeddings

try:
    df_meta, faiss_index, embeddings_array = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# ====== embed model loader (SentenceTransformer optional) ======
@st.cache_resource(show_spinner=False)
def load_embed_model(name=EMBED_MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)
    except Exception:
        return None

embed_model = load_embed_model()

# === LIGHT INTENT GUARD ===
# def shouldUseRAG(user_input: str) -> bool:
#     polite_words = [
#         "terima kasih", "thanks", "makasih", "thx",
#         "good bot", "keren banget", "mantap", "nice", "sip", "oke banget",
#         "udah cukup", "sudah cukup", "makasih ya"
#     ]
#     low = user_input.lower()
#     return any(w in low for w in polite_words)
    
def classify_intent(msg: str) -> str:
    m = msg.lower()

    # polite / goodbye / compliment / non recipe closure
    polite_words = ["thank you", "makasih", "makasi", "terimakasih","terima kasih", "terima kasih ya" "thanks", "mantap", "good job", "keren", "wow", "hebat"]
    for p in polite_words:
        if p in m:
            return "polite"

    # recipe indicators
    recipe_words = ["resep", "ingredients", "bahan", "cara masak", "cook", "masak", "gimana masaknya", "gimana bikinnya"]
    for w in recipe_words:
        if w in m:
            return "recipe"

    # fallback
    return "general"


# ====== helpers ======
def clean_text_for_ui(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("--", "\n").strip()

def embed_query_real(text: str) -> np.ndarray:
    v = embed_model.encode([text], convert_to_numpy=True)[0].astype("float32")
    # normalize
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v

def embed_query_pseudo(text: str, dim=EMBED_DIM) -> np.ndarray:
    import hashlib
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rs = np.random.RandomState(seed)
    v = rs.randn(dim).astype("float32")
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def get_query_vector(text: str) -> np.ndarray:
    if embed_model is not None:
        try:
            return embed_query_real(text)
        except Exception:
            return embed_query_pseudo(text)
    else:
        return embed_query_pseudo(text)

# Cosine similarity helper
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# MMR reranking (diversity)
def mmr_rerank(query_vec: np.ndarray, candidate_indices: List[int], candidate_embeddings: np.ndarray, k: int, lambda_param=MMR_LAMBDA) -> List[int]:
    # candidate_embeddings: matrix with rows matching candidate_indices order
    selected = []
    candidate_sim = np.array([cosine(query_vec, e) for e in candidate_embeddings])
    avail = set(range(len(candidate_indices)))
    if len(candidate_indices) == 0:
        return []
    # pick highest first
    first = int(np.argmax(candidate_sim))
    selected.append(candidate_indices[first])
    avail.remove(first)
    while len(selected) < k and avail:
        best_score = -1e9
        best_idx = None
        for i in list(avail):
            sim_q = candidate_sim[i]
            # max similarity to any selected
            sims_sel = [cosine(candidate_embeddings[i], candidate_embeddings[candidate_indices.index(s_idx)]) for s_idx in selected] if selected else [0]
            max_sim_sel = max(sims_sel) if sims_sel else 0.0
            score = lambda_param * sim_q - (1 - lambda_param) * max_sim_sel
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(candidate_indices[best_idx])
        avail.remove(best_idx)
    return selected

# Retrieval: FAISS search + MMR
def retrieve_and_rerank(query: str, top_k_retrieve=TOP_K_RETRIEVE, top_k_context=TOP_K_CONTEXT):
    qv = get_query_vector(query).astype("float32")
    distances, indices = faiss_index.search(np.array([qv]), top_k_retrieve)
    inds = [int(i) for i in indices[0] if i != -1]
    if len(inds) == 0:
        return []
    # get embeddings for candidate indices (if available)
    if embeddings_array is not None:
        cand_embs = embeddings_array[inds]
    else:
        # if full embeddings not available, approximate with embedding of metadata text (may be slower)
        cand_embs = []
        for i in inds:
            txt = str(df_meta.loc[i, "combined_text"] if "combined_text" in df_meta.columns else df_meta.loc[i, "Title"])
            cand_embs.append(get_query_vector(txt))
        cand_embs = np.vstack(cand_embs).astype("float32")
    # MMR
    selected_indices = mmr_rerank(qv, inds, cand_embs, k=top_k_context)
    # prepare result dicts
    results = []
    for idx in selected_indices:
        row = df_meta.iloc[idx]
        results.append({
            "id": int(idx),
            "title": row.get("Title", ""),
            "ingredients": clean_text_for_ui(row.get("Ingredients", "")),
            "steps": clean_text_for_ui(row.get("Steps", "")),
            "combined_text": row.get("combined_text", "") if "combined_text" in row else "",
        })
    return results

# ====== Ollama wrapper (with safety prompt) ======
use_ollama = True
try:
    import ollama
    use_ollama = True
except Exception as e:
    st.warning(f"Ollama import error: {e}")
    use_ollama = False

RECIPE_SYSTEM_PROMPT = """
#ROLE
You are a highly experienced, top-class chef who can provide recipes with great accuracy according to user requests.
Your role is very important and appreciated by users. Both I and users greatly appreciate your help and recommendations.

#Task
Please explain what you can do if asked by users.

1. Use this format if asked to provide or recommend a recipe:
Cooking Name:
Ingredients:
Cooking Method:

else, just answer normally.
2. Use casual Indonesian like a home food blogger.
3. Use the context of the recipe provided as a reference. Don't make up strange ingredients.
4. Don't repeat the same steps; don't list them repeatedly.
5. You may use the context directly to provide the requested recipe, but revise it in the format provided above.
6. If the ingredient information is insufficient, ask for brief clarification; don't be delusional.

#Limitations
1. You don't need to answer or do anything outside your role.
2. You don't need to explain your reasoning; just answer the final result.
3. You must answer based on the data provided to you from the training data. If the required data is missing, use a fallback response.

#Example Answer
Example 1:
User: Hi, give me a recipe using corn.
Cooking Name: Crispy Corn Fritters
Ingredients:
- 2 kernels of sweet corn kernels
- 3 tablespoons of flour
- 1 tablespoon of cornstarch
- 2 cloves of garlic, finely chopped
- 2 stalks of spring onions, sliced
- 1 egg
- Salt, pepper, and bouillon powder to taste
- Oil for frying
Cooking Instructions:
Mix all ingredients together until well combined → Heat oil → Take 1 tablespoon of batter → Fry until browned and crispy → Remove/drain → Ready to serve.

Example 2:
Cooking Name: Padang-Style Spicy Potatoes
Ingredients:
- 3 medium-sized potatoes
- 3 shallots
- 2 cloves garlic
- 5 curly red chilies (you can add bird's eye chilies if you like it spicy)
- 1 tomato
- Salt, sugar, and stock powder to taste
- Cooking oil
Cooking Instructions:
Cube the potatoes → Fry until half dry → Blend the spices → Stir-fry until fragrant → Add the blended tomatoes → Season to taste → Add the potatoes → Stir well until the spices are absorbed.

Example 3:
Cooking Name: Chicken with Mushrooms in Oyster Sauce
Ingredients:
- 250g chicken breast, diced
- 150g button mushrooms/champignons, sliced
- 3 cloves of minced garlic
- 1 tbsp oyster sauce
- 1 tsp soy sauce
- 1 tsp sweet soy sauce
- Salt, pepper, and stock to taste
- A little water and oil
Cooking Instructions:
Sauté the onion until fragrant → add the chicken → stir until it changes color → add the mushrooms → add the oyster sauce, soy sauce, sweet soy sauce, and a little water → season with salt and pepper → cook until the sauce is absorbed and the sauce has reduced slightly.
"""

def call_ollama(user_query: str, contexts: List[Dict], model: str = MODEL_OLLAMA) -> str:
    # Build safe system+user prompt
    context_snips = []
    for c in contexts:
        s = f"Title: {c['title']}\nIngredients:\n{c['ingredients'][:400]}\nSteps:\n{c['steps'][:800]}"
        context_snips.append(s)
    context_text = "\n\n---\n\n".join(context_snips)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[truncated]"

    prompt = RECIPE_SYSTEM_PROMPT + f"\n\nContext Referensi:\n{context_text}\n\nUser:\n{user_query}"

    if use_ollama:
        try:
            resp = ollama.generate(model=model, prompt=prompt, stream=False)
            resp_txt = resp["response"]
            # remove think tag typical deepseek
            if "<think>" in resp_txt:
                import re
                resp_txt = re.sub(r"<think>.*?</think>", "", resp_txt, flags=re.DOTALL)

            resp_txt = resp_txt.strip()
            return resp_txt
        except Exception as e:
            st.error(f"Ollama gagal dipanggil: {e}")
            use_sim = True
    else:
        use_sim = True

    if use_sim:
        return "Error, chatbot tidak jalan"

# ====== Streamlit chat UI (bubble via st.chat_message) ======
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role": "user"/"assistant", "content": "...", "retrieval": [...]}

with st.sidebar:
    st.header("Pengaturan & Info")
    st.write("Model (local):", MODEL_OLLAMA)
    st.write("Embedding model:", EMBED_MODEL_NAME)
    st.write("Retrieval initial k:", TOP_K_RETRIEVE)
    st.write("MMR lambda:", MMR_LAMBDA)
    st.divider()
    st.write("Files used:")
    st.write(f"- {METADATA_CSV}")
    st.write(f"- {FAISS_INDEX_PATH}")
    if use_ollama:
        st.success("Ollama client available")
    else:
        st.warning("Ollama client NOT available → simulated replies used")

# Chat input
user_text = st.chat_input("Write ingredients / recipes request... (ex:'Hi, give me a recipe using chicken)')")

# Render past history
for m in st.session_state.history:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(m["content"])
            # show small ref titles if exist
            if m.get("retrieval"):
                titles = "  •  ".join([r["title"] for r in m["retrieval"]])
                st.caption(f"Referensi: {titles}")

# On new user input
if user_text:
    # append user
    st.session_state.history.append({"role": "user", "content": user_text})

    # NEW INTENT CLASSIFIER
    intent = classify_intent(user_text)
    st.session_state.last_intent = intent

    if intent == "polite":
        bot_reply = random.choice([
            "You're welcome~",
            "Alrighty, happy cooking!",
            "Of course, let's go cooking!",
            "Okay, that's all for now!"
        ])
        st.session_state.history.append({"role": "assistant", "content": bot_reply, "retrieval": []})
        st.rerun()

    elif intent == "general":
        use_context = []   # nanti ollama tetap dipanggil tapi tanpa retrieval

    elif intent == "recipe":
        # normal rag workflow
        use_context = retrieve_and_rerank(user_text, top_k_retrieve=TOP_K_RETRIEVE, top_k_context=TOP_K_CONTEXT)


    # retrieval (existing)
    assistant_text = call_ollama(user_text, use_context, model=MODEL_OLLAMA)
    assistant_text = assistant_text.replace("--", "\n")
    st.session_state.history.append({"role": "assistant", "content": assistant_text, "retrieval": use_context})
    st.rerun()

# Quick action buttons
cols = st.columns([1,1,1,1])

with cols[1]:
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
