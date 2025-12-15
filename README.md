---
title: knowledge-retrieval-assistant
app_file: app_gradio.py
sdk: gradio
sdk_version: 6.1.0
---
# Web Q&A Dokumen (TF-IDF)

Aplikasi web sederhana berbasis **Flask** untuk melakukan **pencarian jawaban** dari dokumen **PDF/TXT** menggunakan **TF-IDF + cosine similarity**.
Workflow-nya:

1. Upload dokumen (PDF atau TXT)
2. Sistem preprocessing (bersihkan teks, buang header/footer berulang, pecah paragraf, chunking)
3. Sistem membangun indeks TF-IDF
4. Kamu ketik pertanyaan → aplikasi mengambil potongan teks (chunk) yang paling relevan beserta nomor halamannya

---

## Fitur

* ✅ Upload dokumen **PDF** atau **TXT**
* ✅ Normalisasi teks (hapus spasi aneh, baris kosong berlebih, dll.)
* ✅ Deteksi & penghapusan **header/footer berulang** per halaman
* ✅ Pemecahan ke **paragraf** dan **chunk** (sliding window dengan overlap)
* ✅ Pembuatan **indeks TF-IDF** (`sklearn`), disimpan ke folder `index_output/`
* ✅ Pencarian berbasis **cosine similarity** untuk pertanyaan user
* ✅ Menampilkan:

  * Potongan teks (chunk) paling relevan
  * Skor similarity
  * Nomor halaman terkait
* ✅ Logging Q&A ke file `qa_log.json` (opsional, untuk analisis kemudian)

---

## Tech Stack

* **Backend**: [Flask](https://flask.palletsprojects.com/)
* **Frontend**: HTML + [Bootstrap 5](https://getbootstrap.com/)
* **NLP / IR**:

  * `pdfplumber` – ekstraksi teks dari PDF
  * `scikit-learn` – `TfidfVectorizer`, cosine similarity
  * `Sastrawi` – stopword & stemming bahasa Indonesia (jika terpasang)
  * `numpy`, `scipy`

---

## Struktur Project

Kurang lebih seperti ini:

```text
.
├─ app.py
├─ templates/
│  └─ index.html
├─ index_output/           # dibuat otomatis saat build indeks
│  ├─ vectorizer.pkl
│  ├─ tfidf_matrix.npz
│  └─ chunk_meta.json
├─ qa_log.json             # dibuat otomatis saat ada pertanyaan
└─ README.md
```

> Catatan: `index_output/` dan `qa_log.json` akan muncul setelah kamu menjalankan aplikasi dan memproses dokumen/bertanya.

---

## Prasyarat

* Python **3.8+** (disarankan 3.9 atau lebih baru)
* `pip`
* (Opsional tapi enak) Virtual environment (`venv`, `conda`, dll.)

---

## Instalasi

1. **Clone / copy** project ini

   ```bash
   git clone <url-repo-ini>
   cd <nama-folder-repo>
   ```

2. **(Opsional) Buat virtualenv**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate     # Windows
   ```

3. **Install dependensi**

   Langsung via `pip`:

   ```bash
   pip install flask pdfplumber scikit-learn numpy scipy Sastrawi
   ```

   Atau kalau kamu pakai `requirements.txt`, isi dengan misalnya:

   ```text
   flask
   pdfplumber
   numpy
   scipy
   scikit-learn
   Sastrawi
   ```

   lalu:

   ```bash
   pip install -r requirements.txt
   ```

---

## Cara Menjalankan

Jalankan aplikasi Flask:

```bash
python app.py
```

Secara default, aplikasi akan berjalan di:

```text
http://localhost:5000
```

Buka URL itu di browser.

> Catatan: Untuk production, **jangan** pakai `debug=True` dan sebaiknya pakai WSGI server seperti `gunicorn` di belakang Nginx atau sejenisnya.

---

## Cara Pakai

1. Buka `http://localhost:5000`
2. Di bagian **"Upload dokumen (PDF / TXT)"**:

   * Pilih file `.pdf` atau `.txt`
   * Klik **"Proses Dokumen"**
3. Setelah berhasil:

   * Akan muncul info:

     * Nama dokumen
     * Jumlah halaman, paragraf, dan chunk
     * Preview teks dokumen
4. Di bagian **"Ajukan pertanyaan"**:

   * Ketik pertanyaan terkait isi dokumen
   * Klik **"Cari Jawaban"**
5. Di bawahnya akan tampil:

   * Beberapa hasil (chunk) dengan:

     * Rank
     * Skor similarity
     * Nomor halaman
     * Potongan teks relevan

Kalau belum upload dokumen dan kamu coba tanya, aplikasi akan ngasih pesan error “Belum ada dokumen yang di-upload”.

---

## Konfigurasi Penting (di `app.py`)

Beberapa parameter bisa kamu tweak sesuai kebutuhan:

```python
MIN_PARA_LEN = 20             # panjang minimal paragraf (karakter)
MAX_PARAS_PER_CHUNK = 3       # jumlah paragraf per chunk
CHUNK_OVERLAP = 1             # overlap antar chunk (dalam paragraf)
HEADER_RATIO = 0.3            # threshold deteksi header/footer berulang
LOG_PATH = "qa_log.json"      # lokasi file log
```

* **MIN_PARA_LEN**
  Besar → paragraf lebih panjang (lebih sedikit, tapi lebih padat).
  Kecil → paragraf lebih pendek (lebih banyak, tapi bisa lebih noisy).

* **MAX_PARAS_PER_CHUNK & CHUNK_OVERLAP**
  Mengatur ukuran jendela teks:

  * Naikkan `MAX_PARAS_PER_CHUNK` kalau kamu ingin konteks lebih panjang.
  * Naikkan `CHUNK_OVERLAP` kalau kamu ingin transisi antar chunk lebih halus.

* **HEADER_RATIO**
  Dipakai untuk mengenali baris header/footer yang sering muncul di banyak halaman, lalu dibuang.

---

## Batasan

* Sistem **tidak “mengerti”** arti secara mendalam seperti LLM; ini pure **retrieval TF-IDF**.
* Jawaban yang muncul berupa **potongan teks** relevan, bukan kalimat baru hasil generasi.
* Kualitas hasil sangat tergantung:

  * kualitas dokumen (rapi/tidak)
  * bahasa yang konsisten (ID/EN)
  * jumlah noise (scan buruk, banyak simbol, dll.)
* Untuk dokumen yang **sangat besar**, proses build indeks bisa butuh waktu dan RAM lebih.

---

## Ide Pengembangan Lanjut

Kalau mau naik level lagi, beberapa opsi:

* Tambah **evaluasi** berbasis label (misalnya upload CSV pertanyaan–jawaban dan hitung NDCG/MRR).
* Gabung dengan **LLM** (misalnya lewat API) untuk melakukan **answer generation** dari chunk yang diambil TF-IDF.
* Tambah opsi:

  * pilihan jumlah hasil (top-k)
  * filter per halaman
  * download log Q&A dalam bentuk CSV.
* Migrasi UI ke:

  * [Streamlit](https://streamlit.io/)
  * atau [Gradio](https://www.gradio.app/)
    biar bisa deploy cepat ke HuggingFace Spaces / Render / dsb.

---

## Lisensi

Silakan gunakan dan modifikasi sesuka hati.
Kalau mau formal, tambahkan bagian lisensi (MIT/BSD/GPL, dll.) sesuai kebutuhan project-mu.
