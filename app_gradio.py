import os
import re
import json
from io import BytesIO
from collections import Counter
from datetime import datetime

import numpy as np
import pdfplumber
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Opsional: Sastrawi untuk stopword & stemming bahasa Indonesia
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    stopwords_ind = set(StopWordRemoverFactory().get_stop_words())
    stemmer = StemmerFactory().create_stemmer()
except Exception:
    stopwords_ind = set()
    stemmer = None

# ============================
# Konfigurasi dasar
# ============================
MIN_PARA_LEN = 20
MAX_PARAS_PER_CHUNK = 3
CHUNK_OVERLAP = 1
HEADER_RATIO = 0.3

LOG_PATH = "qa_log.json"

# ============================
# Fungsi utility & preprocessing
# ============================


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Baca PDF lalu gabungkan semua halaman + marker [PAGE X].
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


def normalize_text(raw: str) -> str:
    if not raw:
        return ""
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def remove_repeated_header_footer(per_page_texts, min_ratio=HEADER_RATIO):
    if len(per_page_texts) <= 1:
        return per_page_texts

    top_lines, bottom_lines = [], []
    for p in per_page_texts:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        top_lines.extend(lines[:3])
        bottom_lines.extend(lines[-3:])

    total = max(1, len(per_page_texts))
    top_rm = {ln for ln, c in Counter(top_lines).items() if c / total >= min_ratio}
    bottom_rm = {
        ln for ln, c in Counter(bottom_lines).items() if c / total >= min_ratio
    }

    cleaned = []
    for p in per_page_texts:
        lines = p.splitlines()
        new_lines = []
        for i, ln in enumerate(lines):
            s = ln.strip()
            if (i < 4 and s in top_rm) or (len(lines) - i <= 4 and s in bottom_rm):
                continue
            new_lines.append(ln)
        cleaned.append("\n".join(new_lines))

    return cleaned


def split_into_paragraphs_from_page(page_text, min_len=MIN_PARA_LEN):
    t = re.sub(r"\n[ \t]*\n+", "\n\n", page_text.strip())
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if len(p.strip()) >= min_len]
    if not paras:
        # fallback: gabung beberapa line jadi paragraf
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        cur, paras = [], []
        for ln in lines:
            cur.append(ln)
            if len(" ".join(cur)) >= min_len:
                paras.append(" ".join(cur))
                cur = []
        if cur:
            paras.append(" ".join(cur))
    return paras


def make_chunks(paragraphs, per_chunk=MAX_PARAS_PER_CHUNK, overlap=CHUNK_OVERLAP):
    """
    paragraphs: list of (page_no, paragraf_text)
    """
    chunks = []
    i = 0
    while i < len(paragraphs):
        window = paragraphs[i : i + per_chunk]
        chunk_text = "\n".join([p[1] for p in window])
        pages = sorted(list({p[0] for p in window}))
        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": chunk_text,
                "pages": pages,
            }
        )
        i += max(1, per_chunk - overlap)
    return chunks


def preprocess_for_tfidf(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    toks = t.split()
    if stopwords_ind:
        toks = [w for w in toks if w not in stopwords_ind]
    if stemmer:
        toks = [stemmer.stem(w) for w in toks]
    return " ".join(toks)


def preprocess_document(
    full_text: str,
    min_para_len: int = MIN_PARA_LEN,
    max_paras_per_chunk: int = MAX_PARAS_PER_CHUNK,
    overlap: int = CHUNK_OVERLAP,
    header_ratio: float = HEADER_RATIO,
):
    # Normalisasi
    norm = normalize_text(full_text)

    # Split halaman berdasarkan marker [PAGE X]
    pages = re.split(r"\[PAGE\s*\d+\]", norm)
    pages = [p.strip() for p in pages if p.strip()]
    if len(pages) <= 1 and norm:
        # fallback kalau tidak ada marker page
        pages = [norm[i : i + 1500] for i in range(0, len(norm), 1500)]

    # Hapus header/footer berulang
    pages = (
        remove_repeated_header_footer(pages, min_ratio=header_ratio) if pages else []
    )

    # Split paragraf
    paragraphs = []
    for page_no, ptext in enumerate(pages, start=1):
        paras = split_into_paragraphs_from_page(ptext, min_len=min_para_len)
        for para in paras:
            paragraphs.append((page_no, para))

    # Dedup sederhana
    seen = set()
    dedup_paragraphs = []
    for pg, para in paragraphs:
        key = para.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup_paragraphs.append((pg, para))

    # Chunking
    chunks = make_chunks(
        dedup_paragraphs, per_chunk=max_paras_per_chunk, overlap=overlap
    )

    # Tambah processed_text
    for c in chunks:
        c["processed_text"] = preprocess_for_tfidf(c["text"])

    result = {
        "pages": len(pages),
        "paragraphs": len(dedup_paragraphs),
        "chunks": len(chunks),
        "processed_chunks": chunks,
    }
    return result


def build_tfidf_index(processed_chunks, max_features=12000, ngram_range=(1, 2)):
    texts = [c.get("processed_text", "") for c in processed_chunks]
    meta = [
        {"chunk_id": c.get("chunk_id"), "pages": c.get("pages"), "raw": c.get("text")}
        for c in processed_chunks
    ]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # opsional: simpan ke disk
    os.makedirs("index_output", exist_ok=True)
    import pickle

    with open("index_output/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz("index_output/tfidf_matrix.npz", tfidf_matrix)
    with open("index_output/chunk_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[build] TF-IDF built: {len(texts)} chunks, matrix shape {tfidf_matrix.shape}"
    )
    return vectorizer, tfidf_matrix, texts, meta


def query_tfidf(vectorizer, tfidf_matrix, texts, meta, query, top_k=3):
    q = preprocess_for_tfidf(query)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for idx in idxs:
        results.append(
            {
                "chunk_idx": int(idx),
                "chunk_id": int(meta[idx]["chunk_id"]),
                "score": float(sims[idx]),
                "pages": meta[idx]["pages"],
                "raw": meta[idx]["raw"],
            }
        )
    return results


# ============================
# Fungsi untuk Gradio
# ============================


def handle_upload(file_path):
    """
    Dipanggil saat user upload file di Gradio.
    Mengembalikan:
    - info dokumen (Markdown)
    - preview teks
    - vectorizer, matrix, texts, meta, nama dokumen (disimpan di gr.State)
    """
    if not file_path:
        return (
            "❌ Belum ada file yang di-upload.",
            "",
            None,
            None,
            None,
            None,
            None,
        )

    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text.strip():
        return (
            "❌ File kosong atau tidak bisa diekstrak teksnya.",
            "",
            None,
            None,
            None,
            None,
            filename,
        )

    processed = preprocess_document(text)
    vectorizer, tfidf_matrix, texts, meta = build_tfidf_index(
        processed["processed_chunks"]
    )

    stats_line = (
        f"- Halaman: **{processed['pages']}**\n"
        f"- Paragraf unik: **{processed['paragraphs']}**\n"
        f"- Chunk: **{processed['chunks']}**"
    )

    info_md = f"""### Dokumen terproses

**Nama file**: `{filename}`  

{stats_line}
"""

    preview = text[:2000]

    return (
        info_md,
        preview,
        vectorizer,
        tfidf_matrix,
        texts,
        meta,
        filename,
    )


def answer_question(question, vectorizer, tfidf_matrix, texts, meta, docname):
    question = (question or "").strip()
    if not question:
        return "⚠️ Pertanyaan tidak boleh kosong."

    if vectorizer is None or tfidf_matrix is None or texts is None or meta is None:
        return "⚠️ Belum ada dokumen yang di-upload. Upload dulu PDF/TXT-nya."

    results = query_tfidf(vectorizer, tfidf_matrix, texts, meta, question, top_k=5)
    if not results:
        return "Tidak ada hasil yang relevan ditemukan."

    # Logging ke file (opsional)
    try:
        log_item = {
            "time": datetime.now().isoformat(),
            "question": question,
            "answers": results,
            "document": docname,
        }
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lg = json.load(f)
        else:
            lg = []
        lg.append(log_item)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(lg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Gagal menyimpan log:", e)

    md = f"### Hasil untuk pertanyaan: `{question}`\n\n"
    if docname:
        md += f"**Dokumen**: `{docname}`\n\n"

    for i, r in enumerate(results, start=1):
        pages = ", ".join(str(p) for p in r["pages"])
        snippet = r["raw"].strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."

        md += f"**Rank {i}** — skor: `{r['score']:.4f}` — halaman: {pages}\n\n"
        md += f"> {snippet}\n\n"

    return md


# ============================
# Definisi UI Gradio
# ============================

with gr.Blocks(title="Aplikasi Knowledge Retrieval Assistant - Kelompok A") as demo:
    gr.Markdown(
        """
    # Aplikasi Knowledge Retrieval Assistant – Kelompok A

    Upload dokumen **PDF / TXT**, lalu ajukan pertanyaan.  
    Sistem akan mencari potongan teks paling relevan menggunakan **TF-IDF + cosine similarity**.
    """
    )

    # State untuk menyimpan indeks & metadata
    state_vectorizer = gr.State()
    state_matrix = gr.State()
    state_texts = gr.State()
    state_meta = gr.State()
    state_docname = gr.State()

    with gr.Row():
        file_input = gr.File(
            label="Upload dokumen (PDF / TXT)",
            file_types=[".pdf", ".txt"],
            type="filepath",
        )
        doc_info = gr.Markdown("Belum ada dokumen yang di-upload.")

    preview_box = gr.Textbox(
        label="Preview teks dokumen",
        lines=12,
        interactive=False,
    )

    gr.Markdown("---")

    question_box = gr.Textbox(
        label="Pertanyaan",
        placeholder="Tulis pertanyaan terkait isi dokumen di sini...",
        lines=2,
    )
    ask_btn = gr.Button("Cari Jawaban")
    answer_box = gr.Markdown()

    # Event: saat file berubah
    file_input.change(
        fn=handle_upload,
        inputs=file_input,
        outputs=[
            doc_info,
            preview_box,
            state_vectorizer,
            state_matrix,
            state_texts,
            state_meta,
            state_docname,
        ],
    )

    # Event: saat klik tombol "Cari Jawaban"
    ask_btn.click(
        fn=answer_question,
        inputs=[
            question_box,
            state_vectorizer,
            state_matrix,
            state_texts,
            state_meta,
            state_docname,
        ],
        outputs=answer_box,
    )


if __name__ == "__main__":
    # share=True kalau mau gampang diakses dari device lain saat dev
    demo.launch(share=True, debug=True, pwa=True)
