# 🧾 SmartSplit Bill AI

> AI-powered receipt scanner and bill splitter — scan any receipt, split it fairly, settle instantly.

**Data Science & ML Bootcamp — Dibimbing**  
**Assignment: Mini Project Product AI**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Installation & Running](#-installation--running)
3. [Usage Flow](#-usage-flow)
4. [AI Model Research](#-ai-model-research)
5. [Example Results](#-example-results)
6. [Model Comparison Analysis](#-model-comparison-analysis)
7. [Product Evaluation & Analysis](#-product-evaluation--analysis)

---

## 🚀 Project Overview

SmartSplit Bill AI is a Streamlit web application that solves the common problem of splitting restaurant or shopping bills among a group of people. The app uses AI to:

1. **Scan** a receipt photo and extract all item data (name, qty, price)
2. **Review** the extracted data and correct any errors
3. **Split** the bill by assigning each item to a person
4. **Summarize** the exact amount each person owes

### Supported AI Models

| Model          | Type                         | Runs On                         |
| -------------- | ---------------------------- | ------------------------------- |
| Gemini 3 Flash | Multimodal LLM (API)         | Google Cloud (requires API key) |
| Donut CORD v2  | OCR-free Transformer (local) | Your machine (CPU/GPU)          |

---

## 🛠️ Installation & Running

### Prerequisites

- Python 3.10 or higher
- A Gemini API key ([get one here](https://aistudio.google.com/app/apikey)) — **if using Gemini model**

### Option A: Run Locally (Recommended)

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/smartsplit-bill-ai.git
cd smartsplit-bill-ai
```

**2. Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

For Gemini only (lighter, no GPU needed):

```bash
pip install streamlit Pillow pandas python-dotenv google-generativeai
```

For all models including Donut (requires ~800 MB download on first run):

```bash
pip install -r requirements.txt
```

**4. Set your Gemini API key**

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and paste your key:
# GEMINI_API_KEY=AIzaSy...
```

**5. Run the app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

### Option B: Run with Docker

```bash
# Build the image
docker build -t smartsplit-bill-ai .

# Run the container (replace YOUR_KEY with your Gemini API key)
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=YOUR_KEY \
  smartsplit-bill-ai
```

Open `http://localhost:8501` in your browser.

---

## 🗺️ Usage Flow

The app guides you through **5 steps**:

```
📸 Upload  →  📊 Review  →  👥 People  →  🎯 Assign  →  💰 Summary
```

| Step           | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| **1. Upload**  | Take or upload a photo of your receipt. Select the AI model and click "Scan".      |
| **2. Review**  | Check the extracted items. Correct any errors using the editable table.            |
| **3. People**  | Enter the names of everyone splitting the bill (min. 1 person).                    |
| **4. Assign**  | Select who pays for each item. Additional charges (tax, service) split equally.    |
| **5. Summary** | View each person's total. Verify the grand total matches the bill. Export as JSON. |

> 💡 **Tip:** Use "⚡ Split Everything Equally" in Step 4 to auto-distribute items evenly.

---

## 🔬 AI Model Research

### Models Evaluated

#### Model 1: Gemini 3 Flash (Google)

- **Architecture:** Gemini multimodal LLM (decoder-only transformer)
- **Type:** API-based, closed-source
- **Approach:** OCR-free — raw image pixels + prompt → structured JSON output
- **HuggingFace:** N/A (Google API)
- **Paper:** Gemini: A Family of Highly Capable Multimodal Models (2023)
- **Key strength:** Multilingual, handles varied receipt formats and Indonesian text natively

#### Model 2: Donut CORD v2 (Naver CLOVA)

- **Architecture:** VisionEncoderDecoder (Swin Transformer encoder + BART decoder)
- **Type:** Local model (no API, runs on your machine)
- **Approach:** OCR-free — end-to-end document understanding without any OCR pre-processing
- **HuggingFace:** `naver-clova-ix/donut-base-finetuned-cord-v2`
- **Paper:** OCR-Free Document Understanding Transformer, Kim et al. (2022)
- **Key strength:** Receipt-specific model, no internet required, privacy-friendly

---

## 📸 Example Results

> **⚠️ Instructions for student:** Replace the placeholders below with your actual results.
> Include the receipt image file in a folder called `images/` in the repo.

---

### Bill 1: [e.g., Struk Restoran ABC]

**Input Image:**

> 📌 _Add image here: `![Bill 1](images/bill1.jpg)`_

**Gemini 3 Flash — Raw JSON Output:**

```json
{
  "items": [{ "name": "[isi dengan hasil ekstraksi]", "qty": 1, "unit_price": 0, "total_price": 0 }],
  "subtotal": 0,
  "additional_charges": [{ "name": "PPN", "amount": 0 }],
  "total": 0
}
```

**Donut CORD v2 — Raw JSON Output:**

```json
{
  "items": [{ "name": "[isi dengan hasil ekstraksi]", "qty": 1, "unit_price": 0, "total_price": 0 }],
  "subtotal": 0,
  "additional_charges": [],
  "total": 0
}
```

---

### Bill 2: [e.g., Struk Supermarket XYZ]

**Input Image:**

> 📌 _Add image here: `![Bill 2](images/bill2.jpg)`_

**Gemini 3 Flash — Raw JSON Output:**

```json
{
  "items": [{ "name": "[isi dengan hasil ekstraksi]", "qty": 1, "unit_price": 0, "total_price": 0 }],
  "subtotal": 0,
  "additional_charges": [],
  "total": 0
}
```

**Donut CORD v2 — Raw JSON Output:**

```json
{
  "items": [{ "name": "[isi dengan hasil ekstraksi]", "qty": 1, "unit_price": 0, "total_price": 0 }],
  "subtotal": 0,
  "additional_charges": [],
  "total": 0
}
```

---

## 📊 Model Comparison Analysis

### Experiment Setup

| Parameter   | Value                                                   |
| ----------- | ------------------------------------------------------- |
| Device      | _[isi: e.g., MacBook Air M2 / ASUS TUF A15 RTX 3060]_   |
| OS          | _[isi: e.g., macOS 14.4 / Windows 11]_                  |
| Python      | 3.11                                                    |
| PyTorch     | _[isi versi, contoh: 2.2.0]_                            |
| Test images | 2 Indonesian receipts (restaurant + retail/supermarket) |

---

### Quantitative Comparison

| Criteria                     | Gemini 3 Flash | Donut CORD v2  |
| ---------------------------- | -------------- | -------------- |
| **Inference Speed — Bill 1** | _X.XX sec_     | _X.XX sec_     |
| **Inference Speed — Bill 2** | _X.XX sec_     | _X.XX sec_     |
| **Items Detected (Bill 1)**  | _N / N total_  | _N / N total_  |
| **Items Detected (Bill 2)**  | _N / N total_  | _N / N total_  |
| Internet Required            | ✅ Yes         | ❌ No          |
| GPU Required                 | ❌ No          | ⚠️ Recommended |
| Indonesian Text Support      | ✅ Excellent   | ⚠️ Limited     |

---

### Qualitative Analysis

#### Gemini 3 Flash

**Strengths:**

- _[Tulis observasi kamu, contoh: Akurasi tinggi pada teks Indonesia, bisa membaca font yang tidak standar, mampu mengidentifikasi semua item dengan benar]_
- _[contoh: Response cepat melalui API, tidak perlu GPU lokal]_
- _[contoh: Mampu memahami konteks (misalnya diskon yang dikurangi dari harga)]_

**Weaknesses:**

- _[contoh: Memerlukan koneksi internet dan API key berbayar]_
- _[contoh: Latensi bergantung pada kondisi jaringan]_
- _[contoh: Tidak bisa digunakan offline / mode privat]_

---

#### Donut CORD v2

**Strengths:**

- _[contoh: Berjalan sepenuhnya offline, tidak ada data yang dikirim ke server eksternal]_
- _[contoh: Cepat setelah model dimuat ke memori, terutama dengan GPU]_
- _[contoh: Model spesifik untuk struk/receipt, tidak perlu prompt engineering]_

**Weaknesses:**

- _[contoh: Dilatih pada dataset CORD yang sebagian besar berisi struk Korea, sehingga akurasi pada struk Indonesia lebih rendah]_
- _[contoh: Kesulitan membaca nama item dalam Bahasa Indonesia]_
- _[contoh: Model besar (~800 MB), membutuhkan waktu download dan RAM besar]_

---

### ✅ Chosen Model: Gemini 3 Flash

**Reasoning:**

1. **Higher accuracy on Indonesian receipts** — Gemini is trained on multilingual data and handles Bahasa Indonesia natively, while Donut CORD v2 is primarily trained on Korean receipt data (CORD dataset).

2. **No local hardware requirement** — Gemini runs on Google's infrastructure; Donut requires substantial RAM and benefits heavily from a GPU.

3. **Better handling of varied formats** — Indonesian receipts vary widely in font, layout, and content. Gemini's general-purpose reasoning handles this variability better than a task-specific model.

4. **Simpler deployment** — No large model download needed; the app can be deployed on any server without GPU support.

_[Tambahkan alasan spesifik berdasarkan hasil eksperimen kamu]_

---

## 🧪 Product Evaluation & Analysis

### Qualitative Evaluation

#### A. AI Model (Reading Accuracy)

| Aspect             | Observation                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------- |
| Item name accuracy | _[contoh: Sangat baik untuk struk restoran — semua item terdeteksi dengan nama yang benar]_ |
| Price accuracy     | _[contoh: Akurasi tinggi, hanya ada selisih minor pada harga yang tercetak kecil]_          |
| Additional charges | _[contoh: Pajak dan service charge terdeteksi dengan benar]_                                |
| Inference speed    | _[contoh: ~2.5 detik rata-rata via Gemini API]_                                             |
| Failure cases      | _[contoh: Kesulitan pada struk yang sangat kusut atau tulisan tangan]_                      |

#### B. Web Product

| Aspect              | Observation                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| UI/UX               | _[contoh: Alur 5 langkah intuitif, mudah dipahami pengguna baru]_         |
| Bill split accuracy | _[contoh: Total per orang selalu sama dengan total bill — terverifikasi]_ |
| Performance         | _[contoh: App responsif, tidak ada lag saat mengisi data]_                |
| Edge cases handled  | _[contoh: Pembagian biaya tambahan dengan rounding yang benar]_           |

---

### ⚠️ Identified Weaknesses

#### AI Model Weaknesses

1. **Low-quality image handling** — The model struggles with blurry, dark, or heavily crumpled receipts. There is no image pre-processing step (contrast enhancement, deskew) before sending to the model.

2. **Hallucination risk (Gemini)** — Like all LLMs, Gemini may occasionally invent item names or prices when text is unclear. The current app does not cross-validate extracted data against image features.

3. **Donut Indonesian accuracy** — Donut CORD v2, trained on Korean receipts, misreads Indonesian item names and often fails to detect multi-line item descriptions.

4. **No confidence score** — Neither model provides a confidence score per field, making it difficult to automatically flag uncertain extractions for user review.

#### Product Weaknesses

1. **No image pre-processing** — There is no automatic image enhancement (brightness, contrast, deskew) before extraction, so photo quality heavily affects results.

2. **Single-person item assignment only** — Each item can only be assigned to one person. Splitting a single item among multiple people (e.g., a shared dish) is not yet supported.

3. **No receipt history / session persistence** — Once the browser tab is closed, all data is lost. There is no save/load functionality.

4. **Manual correction friction** — When the AI extracts data incorrectly, users must edit the table cell by cell. A bulk-correction mode would improve UX.

---

### 💡 Improvement Ideas

#### AI Model Improvements

1. **Image pre-processing pipeline** — Add automatic contrast enhancement (`PIL.ImageEnhance`), deskewing, and noise reduction before passing to the model to improve accuracy on low-quality photos.

2. **Fine-tune Donut on Indonesian receipts** — Collect an Indonesian receipt dataset and fine-tune Donut CORD v2 on it to dramatically improve its accuracy for local receipts.

3. **Confidence-based auto-flagging** — Use Gemini's structured output mode with a `confidence` field per extracted item; automatically highlight low-confidence items for user review.

4. **Multi-model ensemble** — Run both Gemini and Donut, then merge their outputs using voting or field-level confidence, improving robustness against model-specific failures.

#### Product Improvements

1. **Item-level splitting** — Allow one item to be split among multiple selected people (e.g., 3 people share a pizza). Calculate proportional cost per person.

2. **Receipt history** — Store past bills in local storage or a database so users can review and re-share previous splits.

3. **Share via link** — Generate a unique URL for each bill split, so participants can view their amount on their own device without needing access to the host's screen.

4. **Payment integration** — Integrate with payment apps (GoPay, OVO, etc.) to generate deep links so each person can pay their share with one tap.

5. **Camera capture** — Add direct camera access (via `st.camera_input`) to eliminate the need to transfer photos from phone to computer before uploading.

---

## 📁 Project Structure

```
smartsplit-bill-ai/
├── app.py                      # Main Streamlit application (5-step wizard)
├── extractor/
│   ├── __init__.py             # Package exports
│   ├── gemini_extractor.py     # Gemini 3 Flash extractor (API)
│   └── donut_extractor.py      # Donut CORD v2 extractor (local)
├── Dockerfile                  # Optional Docker deployment
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

---

## 📜 License

MIT License — free to use, modify, and distribute.
