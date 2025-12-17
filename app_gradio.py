import os
import re
import json
from io import BytesIO
from datetime import datetime

import pdfplumber
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOG_FILE = "knowledge_log.json"


# ============================
# FUNGSI EKSTRAKSI FILE
# ============================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Membaca PDF lalu menggabungkan semua halaman,
    ditambah marker [PAGE X] (kalau mau dipakai nanti).
    Di sini kita pakai full text yang sudah digabung.
    """
    full_text = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            full_text.append(f"[PAGE {i}]\n{text}\n")

    return "\n".join(full_text)


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")


# ============================
# PREPROCESSING SEDERHANA
# ============================

def preprocess_text(text: str) -> str:
    """
    Preprocessing sederhana untuk dokumen SOP & pertanyaan user.
    Sesuai kode asli: tanpa stemming / NLP lanjutan.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================
# FUNGSI BUILD TF-IDF DARI DOKUMEN
# ============================

def build_tfidf_from_document(text: str, segment_length: int = 900, min_seg_len: int = 50):
    """
    - Preprocess dokumen
    - Pecah jadi segmen (documents)
    - Fit TF-IDF
    - Hitung coverage 'akurasi' seperti di kode Colab
    """
    doc_clean = preprocess_text(text)

    # Pecah jadi segmen
    documents = []
    for i in range(0, len(doc_clean), segment_length):
        seg = doc_clean[i : i + segment_length]
        if len(seg.strip()) > min_seg_len:
            documents.append(seg)

    if not documents:
        return None, None, None, "❌ Dokumen terlalu pendek setelah preprocessing, tidak ada segmen yang valid."

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Hitung "akurasi model (coverage dokumen)" seperti di kode
    tfidf_features = set(vectorizer.get_feature_names_out())
    unique_words = set(doc_clean.split())
    covered_words = tfidf_features.intersection(unique_words)

    if len(unique_words) > 0:
        accuracy_percent = round((len(covered_words) / len(unique_words)) * 100, 2)
    else:
        accuracy_percent = 0.0

    stats_md = (
        f"- Jumlah segmen SOP: **{len(documents)}**\n"
        f"- Jumlah kata unik dokumen SOP: **{len(unique_words)}**\n"
        f"- Jumlah fitur TF-IDF: **{len(tfidf_features)}**\n"
        f"- Jumlah kata terwakili TF-IDF: **{len(covered_words)}**\n"
        f"- Akurasi Model (Coverage Dokumen): **{accuracy_percent}%**\n"
    )

    return vectorizer, tfidf_matrix, documents, stats_md


# ============================
# HANDLER UNTUK UPLOAD DI GRADIO
# ============================

def handle_upload(file_path):
    """
    Dipanggil saat user upload file:
    - baca file
    - ekstrak teks (PDF/TXT)
    - build TF-IDF
    - kembalikan info dokumen + preview + state (vectorizer, matrix, documents)
    """
    if not file_path:
        return (
            "❌ Belum ada file yang di-upload.",
            "",
            None,
            None,
            None,
            None,
        )

    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    if filename.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_txt(file_bytes)

    if not raw_text.strip():
        return (
            "❌ File kosong atau teks tidak bisa diekstrak.",
            "",
            None,
            None,
            None,
            filename,
        )

    vectorizer, tfidf_matrix, documents, stats_md = build_tfidf_from_document(raw_text)

    if vectorizer is None:
        # gagal bikin segmen
        return (
            stats_md,
            raw_text[:2000],
            None,
            None,
            None,
            filename,
        )

    info_md = f"""### Dokumen ter-proses

**Nama file**: `{filename}`

{stats_md}
"""

    preview = raw_text[:2000]

    return (
        info_md,
        preview,
        vectorizer,
        tfidf_matrix,
        documents,
        filename,
    )


# ============================
# FUNGSI JAWAB PERTANYAAN
# ============================

def answer_question(
    question: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    documents,
    filename: str,
    threshold: float = 0.05,
):
    """
    Tiruan logika original:
    - preprocess pertanyaan
    - transform dengan vectorizer (tanpa fit)
    - cosine similarity
    - ambil segmen terbaik (+ konteks sebelum/sesudah)
    - simpan Knowledge Log
    """
    question = (question or "").strip()
    if not question:
        return "⚠️ Pertanyaan tidak boleh kosong."

    if vectorizer is None or tfidf_matrix is None or documents is None:
        return "⚠️ Belum ada dokumen yang di-upload atau indeks belum dibuat."

    # Preprocess pertanyaan
    question_clean = preprocess_text(question)

    # Transform query
    query_vector = vectorizer.transform([question_clean])

    # Cosine similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    best_index = int(similarity_scores.argmax())
    best_score = float(similarity_scores[best_index])

    # Ambil konteks segmen sebelum & sesudah
    if best_score >= threshold:
        start = max(best_index - 1, 0)
        end = min(best_index + 2, len(documents))
        answer = "\n\n".join(documents[start:end])
    else:
        answer = "Maaf, SOP yang relevan tidak ditemukan."

    # Simpan ke Knowledge Log
    save_knowledge_log(
        question=question,
        answer=answer,
        similarity_score=round(best_score * 100, 2),
        source_segment_index=best_index,
        document=filename,
    )

    # Kembalikan dalam bentuk Markdown
    md = f"### Pertanyaan\n`{question}`\n\n"
    md += f"**Skor Relevansi**: `{round(best_score * 100, 2)}%`\n\n"
    md += "### Jawaban Chatbot\n\n"
    md += f"{answer}\n\n"
    md += f"_Log pencarian disimpan di `{LOG_FILE}`._"

    return md


# ============================
# KNOWLEDGE LOG
# ============================

def save_knowledge_log(
    question: str,
    answer: str,
    similarity_score: float,
    source_segment_index: int,
    document: str,
):
    """
    Simpan log ke knowledge_log.json
    """
    # pastikan file ada
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []

    log_id = len(logs) + 1
    log_entry = {
        "log_id": log_id,
        "timestamp": datetime.now().isoformat(),
        "document": document,
        "question": question,
        "answer": answer,
        "similarity_score": similarity_score,
        "source_segment_index": int(source_segment_index),
    }

    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)


def load_knowledge_log():
    """
    Baca dan tampilkan Knowledge Log sebagai Markdown ringkas
    """
    if not os.path.exists(LOG_FILE):
        return "Belum ada Knowledge Log. Coba ajukan pertanyaan dulu."

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception as e:
        return f"Gagal membaca Knowledge Log: {e}"

    if not logs:
        return "Belum ada Knowledge Log."

    last_logs = logs[-20:]  # ambil 20 terakhir
    md = "### Knowledge Log (Riwayat Pencarian Terakhir)\n\n"

    for log in reversed(last_logs):
        log_id = log.get("log_id", "-")
        ts = log.get("timestamp", "-")
        doc = log.get("document", "-")
        q = log.get("question", "-")
        score = log.get("similarity_score", "-")
        seg_idx = log.get("source_segment_index", "-")

        snippet = (log.get("answer", "") or "").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        md += f"- **Log ID {log_id}** — `{ts}` — Dokumen: `{doc}` — Skor: {score}% — Segmen: {seg_idx}\n"
        md += f"  - Pertanyaan: `{q}`\n"
        if snippet:
            md += f"  - Cuplikan jawaban: _{snippet}_\n"
        md += "\n"

    return md


# ============================
# DEFINISI UI GRADIO
# ============================

with gr.Blocks(title="SOP Q&A TF-IDF – Gradio") as demo:
    gr.Markdown(
        """
    # SOP Q&A berbasis TF-IDF

    Upload dokumen SOP (**PDF / TXT**), lalu ajukan pertanyaan.  
    Sistem akan memecah dokumen menjadi beberapa segmen, membangun model **TF-IDF**, dan mencari segmen paling relevan menggunakan **cosine similarity**.  
    Setiap pencarian akan disimpan sebagai **Knowledge Log**.
    """
    )

    # State untuk model & dokumen
    state_vectorizer = gr.State()
    state_matrix = gr.State()
    state_documents = gr.State()
    state_filename = gr.State()

    with gr.Row():
        file_input = gr.File(
            label="Upload dokumen SOP (PDF / TXT)",
            file_types=[".pdf", ".txt"],
            type="filepath",
        )
        doc_info = gr.Markdown("Belum ada dokumen yang di-upload.")

    preview_box = gr.Textbox(
        label="Preview teks dokumen (potongan awal)",
        lines=12,
        interactive=False,
    )

    gr.Markdown("---")

    question_box = gr.Textbox(
        label="Pertanyaan SOP",
        placeholder="Tulis pertanyaan terkait SOP di sini...",
        lines=2,
    )
    ask_btn = gr.Button("Cari Jawaban")
    answer_box = gr.Markdown()

    with gr.Accordion("Knowledge Log (Riwayat Pencarian)", open=False):
        log_btn = gr.Button("Refresh Knowledge Log")
        log_box = gr.Markdown("Belum ada Knowledge Log.")

    # Event: upload file
    file_input.change(
        fn=handle_upload,
        inputs=file_input,
        outputs=[
            doc_info,
            preview_box,
            state_vectorizer,
            state_matrix,
            state_documents,
            state_filename,
        ],
    )

    # Event: klik "Cari Jawaban"
    ask_btn.click(
        fn=answer_question,
        inputs=[
            question_box,
            state_vectorizer,
            state_matrix,
            state_documents,
            state_filename,
        ],
        outputs=answer_box,
    )

    # Event: refresh log
    log_btn.click(
        fn=load_knowledge_log,
        inputs=None,
        outputs=log_box,
    )


if __name__ == "__main__":
    demo.launch()
