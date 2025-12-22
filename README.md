---
title: knowledge-retrieval-assistant
app_file: app_gradio.py
sdk: gradio
sdk_version: 6.1.0
---
# Knowledge Retrieval Assistant (TF-IDF Q&A)

Aplikasi ini adalah **Knowledge Retrieval Assistant** sederhana untuk mendukung **Knowledge Management System (KMS)**.
Pengguna bisa **upload dokumen pengetahuan (PDF/TXT)** lalu mengajukan pertanyaan. Aplikasi akan mencari potongan teks yang paling relevan menggunakan **TF-IDF + cosine similarity** dan menyimpan riwayat pencarian sebagai **Knowledge Log**.

---

## ✨ Fitur Utama

### 🔍 Fitur AI

* Ekstraksi teks dari **PDF** (via `pdfplumber`) atau **TXT**
* **Preprocessing teks**:

  * normalisasi spasi & line break
  * penghapusan header/footer halaman yang berulang
  * pemecahan paragraf dan chunk (dengan overlap)
* **Representasi vektor TF-IDF** dengan `scikit-learn`
* **Pencarian berbasis cosine similarity**:

  * pertanyaan user diubah ke vektor
  * sistem mengembalikan *top-k* potongan teks (chunk) paling relevan
  * menampilkan skor similarity dan nomor halaman

### 🏛️ Fitur KMS

Aplikasi ini mengimplementasikan beberapa komponen Knowledge Management:

1. **Knowledge Capture**

   * Upload dokumen (**PDF/TXT**)
   * Ekstraksi dan preprocessing teks:

     * pemisahan per halaman
     * deteksi dan penghapusan header/footer berulang
     * pemecahan paragraf & chunk

2. **Knowledge Storage**

   * Menyimpan hasil analisis ke filesystem:

     * `index_output/vectorizer.pkl` – model TF-IDF
     * `index_output/tfidf_matrix.npz` – matriks TF-IDF
     * `index_output/chunk_meta.json` – metadata chunk (teks & nomor halaman)
   * Menyimpan **Knowledge Log** ke file:

     * `knowledge_log.json` – riwayat pertanyaan & cuplikan jawaban

3. **Knowledge Retrieval**

   * Pencarian pengetahuan berbasis pertanyaan user:

     * mengembalikan chunk teks paling relevan
     * menampilkan nomor halaman terkait

4. **Knowledge Sharing**

   * Antarmuka web (Flask atau Gradio) yang:

     * menampilkan ringkasan dokumen (jumlah halaman, paragraf, chunk)
     * menampilkan hasil pencarian untuk pertanyaan pengguna
     * menampilkan **Knowledge Log (riwayat pencarian)** di UI (mode Gradio)

---

## 🧠 Model AI yang Digunakan

Model AI di aplikasi ini adalah kombinasi:

* **TF-IDF (TfidfVectorizer)** dari `scikit-learn`
* **Cosine similarity** untuk mengukur kedekatan antara:

  * vektor pertanyaan
  * vektor chunk dokumen

Ini termasuk dalam kategori:

* **Model local dengan scikit-learn**, dan
* **Contoh sederhana ML (TF-IDF)** seperti yang dipersyaratkan pada deskripsi tugas.

---

## 🧩 Tech Stack

* **Bahasa**: Python 3.8+
* **NLP / IR**:

  * `pdfplumber`
  * `scikit-learn`
  * `numpy`, `scipy`
  * `Sastrawi` (opsional, untuk stopword & stemming Bahasa Indonesia)
* **Web Framework**:

  * **Flask** (Mode 1 – klasik)
  * **Gradio** (Mode 2 – recommended untuk deploy cepat)

---

## 📁 Struktur Project

Struktur direktori (kurang lebih):

```text
.
├─ app.py                # Mode Flask
├─ app_gradio.py         # Mode Gradio (disarankan untuk deployment)
├─ templates/
│  └─ index.html         # Template Flask
├─ index_output/         # Dibuat otomatis (indeks TF-IDF)
│  ├─ vectorizer.pkl
│  ├─ tfidf_matrix.npz
│  └─ chunk_meta.json
├─ knowledge_log.json    # Dibuat otomatis (Knowledge Log)
├─ README.md
└─ requirements.txt
```

> Catatan:
>
> * Folder `index_output/` dan file `knowledge_log.json` akan muncul setelah aplikasi dijalankan dan digunakan.
> * Di mode Gradio, **Knowledge Log** bisa dilihat langsung dari UI.

---

## ⚙️ Instalasi

1. Clone / salin repo:

```bash
git clone https://github.com/wardanaa/knowledge-retrieval-assistant.git
cd knowledge-retrieval-assistant
```

2. (Opsional Tapi Direkomendasikan) Buat virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# atau
venv\Scripts\activate         # Windows
```

3. Install dependensi:

```bash
pip install -r requirements.txt
```

Contoh isi `requirements.txt`:

```text
flask
pdfplumber
numpy
scipy
scikit-learn
Sastrawi
gradio
```

---

## 🚀 Mode Aplikasi

Aplikasi bisa dijalankan dalam dua mode:

* **Mode Gradio** – antarmuka modern, mudah di-deploy (disarankan)
* **Mode Flask** – antarmuka web klasik (HTML + Bootstrap)

### 1️⃣ Mode Gradio (Recommended)

File utama: `app_gradio.py`

Jalankan:

```bash
python app_gradio.py
```

Gradio biasanya akan tampil di:

```text
http://127.0.0.1:7860
```

#### Alur Penggunaan (Gradio)

1. Buka URL Gradio di browser.
2. **Upload dokumen (PDF/TXT)** melalui komponen file.
3. Aplikasi akan:

   * mengekstrak dan memproses teks
   * membangun indeks TF-IDF
   * menampilkan:

     * info dokumen (jumlah halaman, paragraf, chunk)
     * preview teks dokumen
4. Tuliskan **pertanyaan** pada textbox, lalu klik **“Cari Jawaban”**.
5. Aplikasi akan menampilkan:

   * beberapa **Rank** jawaban
   * **skor similarity**
   * **nomor halaman**
   * cuplikan teks jawaban
6. Buka **accordion “Knowledge Log (Riwayat Pencarian)”** untuk melihat:

   * daftar pertanyaan terbaru
   * timestamp
   * dokumen terkait
   * halaman jawaban
   * cuplikan jawaban (snippet)

> **Knowledge Log (Gradio):**
>
> * Disimpan di file `knowledge_log.json`.
> * Ditampilkan langsung di UI lewat tombol **“Refresh Knowledge Log”**.
> * Ini adalah implementasi eksplisit dari *Knowledge Storage* + *Knowledge Retrieval* untuk riwayat pengetahuan.

---

### 2️⃣ Mode Flask (Alternatif)

File utama: `app.py`

Sebelum menjalankan Flask, set `FLASK_SECRET_KEY` di environment agar sesi aman:

macOS / Linux:
```bash
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Windows (PowerShell):
```powershell
$env:FLASK_SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
```

Jalankan:

```bash
python app.py
```

Secara default Flask berjalan di:

```text
http://localhost:5000
```

#### Alur Penggunaan (Flask)

1. Buka `http://localhost:5000`.
2. Di bagian **“Upload dokumen (PDF / TXT)”**:

   * pilih file PDF/TXT
   * klik **“Proses Dokumen”**
3. Setelah berhasil, halaman akan menampilkan:

   * nama dokumen
   * jumlah halaman, paragraf, chunk
   * preview teks
4. Di bagian **“Ajukan pertanyaan”**:

   * tulis pertanyaan terkait isi dokumen
   * klik **“Cari Jawaban”**
5. Aplikasi akan menampilkan daftar hasil dengan:

   * urutan rank
   * skor similarity
   * nomor halaman
   * potongan teks relevan

> Pada mode Flask, Knowledge Log **tetap disimpan** ke `knowledge_log.json`, meskipun belum ditampilkan di UI. File tersebut masih bisa digunakan sebagai bukti komponen **Knowledge Storage** di dokumentasi.

---

## 🧾 Knowledge Log

**File:** `knowledge_log.json`

Setiap kali pengguna mengajukan pertanyaan, aplikasi (mode Gradio & Flask) menyimpan entri log berisi antara lain:

* Waktu (`time`)
* Pertanyaan (`question`)
* Nama dokumen (`document`)
* Halaman jawaban teratas (`top_answer_pages`)
* Cuplikan jawaban teratas (`top_answer_snippet`)

Di mode Gradio:

* Log ini bisa dilihat melalui **accordion “Knowledge Log (Riwayat Pencarian)”**.
* Secara default menampilkan sekitar 20 entri terbaru.

Ini bisa dijadikan bahan:

* Analisis penggunaan sistem
* Sumber pengetahuan tambahan (FAQ internal)
* Bagian pembahasan **Knowledge Management** di laporan.

---

## 📚 Kaitan dengan Knowledge Management System (untuk Laporan)

Ringkasannya:

* **AI Component**

  * TF-IDF + cosine similarity (scikit-learn) sebagai model retrieval.

* **Knowledge Capture**

  * Upload dan ekstraksi dokumen pengetahuan (PDF/TXT).
  * Preprocessing dan chunking sebagai bentuk strukturisasi pengetahuan.

* **Knowledge Storage**

  * Penyimpanan indeks TF-IDF (`index_output/`).
  * Penyimpanan Knowledge Log (`knowledge_log.json`).

* **Knowledge Retrieval**

  * Pencarian chunk relevan dari dokumen berdasarkan pertanyaan user.

* **Knowledge Sharing**

  * Antarmuka web (Flask/Gradio) yang menyajikan:

    * hasil pencarian
    * ringkasan dokumen
    * riwayat pertanyaan (Knowledge Log) ke pengguna.

---

## ⚠️ Batasan

* Sistem hanya melakukan **retrieval**, bukan **jawaban generatif**:

  * output berupa potongan teks original dari dokumen.
* Kualitas hasil sangat dipengaruhi oleh:

  * kualitas dokumen (scan vs teks asli),
  * konsistensi bahasa (ID/EN),
  * jumlah noise.
* Untuk dokumen yang sangat besar, proses indeks bisa memakan waktu & memori lebih besar.

---

## 📝 Lisensi

Silakan gunakan dan modifikasi sesuai kebutuhan.
Tambahkan lisensi formal (misalnya MIT) jika diperlukan untuk keperluan publikasi atau open source.
