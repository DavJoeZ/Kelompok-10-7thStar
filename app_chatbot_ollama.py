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
from gtts import gTTS
import uuid
from streamlit_mic_recorder import mic_recorder
import whisper


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
MODEL_OLLAMA = "qwen2.5:3b"
MAX_CONTEXT_CHARS = 1400
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, "input.wav")


# ====== UI PAGE ======
st.set_page_config(page_title="Chatbot Resep — Ollama", layout="wide")
st.title("🍳 Chatbot Rekomendasi Resep — Ollama (Qwen2.5-3B)")
st.caption("RAG + MMR + persona prompt + FAISS retrieval (local LLM via Ollama).")

# ====== TTS ======
def text_to_speech(text: str):
    tts = gTTS(text=text, lang="id")
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    tts.save(filename)
    return filename

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

@st.cache_resource
def load_whisper():
    return whisper.load_model("small")   # bisa: tiny / base / small

whisper_model = load_whisper()


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

def cleanup_old_tts(keep_last=5):
    audio_msgs = [m for m in st.session_state.history if m.get("audio")]

    for m in audio_msgs[:-keep_last]:
        f = m.get("audio")
        try:
            if f and os.path.exists(f):
                os.remove(f)
        except:
            pass
        m["audio"] = None   # biar UI gak nyoba play lagi

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
Kamu adalah chef kelas atas yang sangat berpengalaman dan bisa memberikan resep masakan dengan sangat akurat sesuai dengan apa yang diminta orang user.
Tugas kamu sangat penting dan dihargai oleh user. Aku dan user sangat menghargai bantuan dan rekomendasimu.

#Tugas
Berikan penjelasan tentang apa yang bisa kamu lakukan jika ditanya user.

1. Gunakan format ini jika diminta untuk memberikan atau merekomendasikan resep
Nama Masakan:
Bahan:
Cara Masak:

2. Gunakan bahasa Indonesia santai seperti food blogger rumahan.
3. Gunakan konteks resep yang diberikan sebagai referensi. Jangan mengarang bahan aneh.
4. Jangan mengulang step yang sama, jangan list berulang.
5. Boleh Gunakan konteksnya secara langsung untuk memberikan resep yang diminta, tapi ubah menjadi format yang diberikan di atas.
6. Jika informasi bahan tidak cukup → tanya klarifikasi singkat, bukan halu.

#Batasan
1. Kamu tidak perlu menjawab atau melakukan hal yang diluar dari tugas kamu
2. Tidak perlu menjelaskan reasoning, langsung jawab hasil final saja.
3. Kamu harus menjawab sesuai dengan data yang diberikan kepadamu dari training data yang ada, jika data yang diperlukan tidak ada, gunakan respon fallback.
4. WAJIB menggunakan Bahasa Indonesia saja. 
5. DILARANG menggunakan bahasa selain Bahasa Indonesia (tidak boleh Mandarin, Inggris, atau bahasa lain).

#Contoh Jawaban
Contoh 1 :
User : Hai, berikan aku resep dengan bahan jagung
Nama Masakan : Perkedel Jagung Crispy
Bahan :
- 2 buah jagung manis pipil
- 3 sdm tepung terigu
- 1 sdm tepung maizena
- 2 siung bawang putih, halus
- 2 batang daun bawang, iris
- 1 butir telur
- garam, lada, kaldu bubuk secukupnya
- minyak untuk menggoreng
Cara Masak:
Campur semua bahan jadi satu sampai rata → panaskan minyak → ambil adonan 1 sdm → goreng hingga kecoklatan dan crispy → angkat/tiriskan → siap disajikan.

Contoh 2 :
Nama Masakan : Kentang Balado Padang
Bahan :
- 3 buah kentang ukuran sedang
- 3 siung bawang merah
- 2 siung bawang putih
- 5 cabai merah keriting (bisa campur rawit kalau mau pedas)
- 1 buah tomat
- garam, gula, kaldu bubuk secukupnya
- minyak goreng
Cara Masak :
Potong kentang dadu → goreng setengah kering → haluskan bumbu → tumis sampai wangi → masukkan tomat blender → bumbui → masukkan kentang → aduk rata sampai bumbu meresap.

Contoh 3 :
Nama Masakan : Ayam Jamur Saus Tiram
Bahan :
- 250 gr dada ayam, potong dadu
- 150 gr jamur kancing / champignon, iris
- 3 siung bawang putih cincang
- 1 sdm saus tiram
- 1 sdt kecap asin
- 1 sdt kecap manis
- garam, lada, kaldu secukupnya
- sedikit air & minyak
Cara Masak:
Tumis bawang sampai wangi → masukkan ayam → aduk sampai berubah warna → masukkan jamur → tambahkan saus tiram, kecap asin, kecap manis, sedikit air → bumbui garam + lada → masak sampai meresap dan kuah agak menyusut.

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
            resp = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "num_predict": 800,   # batas token output
                    "temperature": 0.7
                    },
                stream=False
            )
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

voice_text = None

# Chat input
user_text = st.chat_input("Tulis bahan / permintaan resep... (mis: 'berikan resep dengan bahan ayam')")

st.markdown('<div class="mic-container">', unsafe_allow_html=True)

audio = mic_recorder(
    start_prompt="🎙️",
    stop_prompt="⏹️",
    just_once=True,
    key="mic",
)

st.markdown('</div>', unsafe_allow_html=True)

if audio and "bytes" in audio:
    with open(AUDIO_PATH, "wb") as f:
        f.write(audio["bytes"])

    if os.path.exists(AUDIO_PATH):
        result = whisper_model.transcribe(AUDIO_PATH, language="id")
        voice_text = result["text"]
        st.success(f"🗣️ Kamu bilang: {voice_text}")
    else:
        st.error("File audio gagal disimpan.")


if voice_text:
    user_text = voice_text

# Render past history
for i, m in enumerate(st.session_state.history):
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(m["content"])

            if m.get("audio") and not m.get("played"):
                st.audio(m["audio"], format="audio/mp3", autoplay=True)
                m["played"] = True

            # === TTS BUTTON ===
            if st.button("🔊 Dengarkan", key=f"tts_{i}"):
                audio_file = text_to_speech(m["content"])
                st.audio(audio_file, format="audio/mp3")

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
            "hehe sama sama~",
            "sip, happy cooking!",
            "oke siap, semangat masaknya!",
            "baik, sampai sini dulu ya!"
        ])
        st.session_state.history.append({"role": "assistant", "content": bot_reply, "retrieval": []})
        st.stop()

    elif intent == "general":
        use_context = []   # nanti ollama tetap dipanggil tapi tanpa retrieval

    elif intent == "recipe":
        # normal rag workflow
        use_context = retrieve_and_rerank(user_text, top_k_retrieve=TOP_K_RETRIEVE, top_k_context=TOP_K_CONTEXT)


    # retrieval (existing)
    assistant_text = call_ollama(user_text, use_context, model=MODEL_OLLAMA)
    assistant_text = assistant_text.replace("--", "\n")

    audio_file = text_to_speech(assistant_text)

    st.session_state.history.append({
        "role": "assistant",
        "content": assistant_text,
        "retrieval": use_context,
        "audio": audio_file,
        "played": False
    })
    cleanup_old_tts(keep_last=5)

    st.rerun()

# Quick action buttons
cols = st.columns([1,1,1,1])

with cols[1]:
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
try:
    import ollama
    test = ollama.chat(model=MODEL_OLLAMA, messages=[{"role":"user","content":"ping"}])
    st.write("OLLAMA LIVE ✅")
except Exception as e:
    st.write("OLLAMA FAIL ❌", e)

st.caption("Tips: pastikan embeddings+FAISS dibangun pakai embedding model yang sama seperti EMBED_MODEL_NAME agar retrieval akurat.")
