# AI Chatbot resep makanan

> Membuat AI Chatbot untuk mencari resep makanan dengan bahan yang diberikan oleh user.

## Kelompok 10 - 7Th Star

-   David Joevincent (221110724)
-   Vincent (221113855)
-   Stevie Sawita (221110019)

# Cara Menggunakan

1. Install requirement yang dibutuhkan di `requirement.txt` dengan menjalankan kode ini di terminal

```bash
pip install -r requirements.txt
```

2. Jika file embbed belum ada, jalankan file `embbeding_local.py` untuk membuat `metadata.csv` nya lalu jalankan `Build_faiss_index.py` untuk membuat file `faiss.index` nya
3. Install Ollama di device local anda masing masing, download di link ini https://ollama.com/download
4. Lalu buka CMD seperti powershell dan jalankan kode ini untuk install deepseek secara local

```bash
ollama pull deepseek-r1:1.5b
```

5. Setelah semua sudah terdownload, masuk ke file `app_chatbot_eng.py` dan jalankan kode ini di terminal untuk menjalankan app nya

```bash
python -m streamlit run app_chatbot_eng.py
```

6. Biasanya stremlit akan langsung membuka browser ke halaman aplikasi, jika tidak, masukkan link yang diberikan di terminal untuk membuka aplikasinya di web
