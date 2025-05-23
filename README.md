# ChaAllo (Chat with Allo)

ChAllo adalah chatbot yang dirancang khusus untuk membantu nasabah dan calon nasabah Allo Bank. Fungsinya adalah menjawab pertanyaan umum seputar FAQ Allo Bank yang diambil dari handbook yang sudah disiapkan. Chatbot ini dibangun menggunakan Large Language Model (LLM) OpenAI, sehingga mampu memberikan jawaban yang relevan dan akurat. Topik yang bisa dijawab sangat beragam, mulai dari informasi umum tentang Allo Bank, detail akun, produk, layanan, cara pendaftaran, hingga proses pengaduan nasabah. Data referensi yang digunakan untuk chatbot ini berasal dari hasil scraping (pengambilan data otomatis) dari bagian FAQ di website resmi Allo Bank.

---

## Project Background

Di era digital, nasabah membutuhkan informasi cepat. Namun, layanan pelanggan dari Allo Bank melalui allocare saat ini seringkali kurang optimal. Berdasarkan salah satu ulasan aplikasi yang menyatakan "Email dan whatsapp allocare tidak pernah merespon, harus telepon ke allocare yang setiap komplain pasti habis biaya lebih dari 20 ribu rupiah", terlihat jelas adanya frustrasi akibat respons yang lambat dan biaya komunikasi yang tinggi. Nasabah mengandalkan pencarian manual melalui website atau call center yang memakan waktu dan berpotensi mengurangi kepuasan.

Untuk mengatasi langsung permasalahan ini dan meningkatkan kepuasan nasabah, kami mengembangkan Chatbot ChAllo. Ini adalah asisten otomatis 24/7 yang dirancang untuk menjawab pertanyaan umum berdasarkan FAQ Allo Bank. ChAllo dibangun menggunakan Large Language Model (LLM) OpenAI dan datanya berasal dari web scraping website resmi Allo Bank.

Proyek ini bertujuan meningkatkan kepuasan nasabah (memberikan akses cepat dan tanpa biaya telepon), mengurangi beban operasional bank (meminimalkan pertanyaan berulang), dan meningkatkan efisiensi informasi. Singkatnya, ChAllo akan mentransformasi layanan pelanggan Allo Bank menjadi lebih modern, responsif, dan hemat biaya bagi nasabah.

---

## Objective

Pembuatan Chatbot ChAllo memiliki tujuan utama untuk meningkatkan responsivitas layanan pelanggan dengan menyediakan akses jawaban instan 24/7, sekaligus mengurangi beban dan biaya komunikasi yang selama ini memberatkan nasabah. Kami juga berupaya mengurangi beban operasional Allo Bank dengan meminimalkan pertanyaan umum yang rutin ditangani staf, serta meningkatkan akurasi dan konsistensi informasi yang diberikan melalui sumber data FAQ resmi.

---

## Team Members

| Name                          | Role                 |
|-------------------------------|----------------------|
| Muhammad Faqih Tria Lasamba   | Data Scientist       |
| Ilham Nauval Andrika          | Data Scientist       |
| Nur M Assudais Alkharomain    | Data Analyst         |
| Bagas Distyo Utomo            | Data Engineer        |

---

## File Explanation

| File                           | Description                                     |
|--------------------------------|-------------------------------------------------|
| `Optimized_RAG-gpt4.ipynb`     | RAG model implementation using GPT-4            |
| `RAG_gpt35turbo.ipynb`         | RAG model implementation using GPT-3.5          |
| `RAG_vectorDB.ipynb`           | Vectore store configuration documentation       |
| `allobank_faq_eda.ipynb`       | Exploratory Data Analysis documentation         |
| `app.py`                       | Deployment script for ChAllo chatbot            |
| `faq_allobank.csv`             | RAW dataset from webscraping                    |
| `faq_allobank_clean.csv`       | Cleaned version of RAW dataset                  |
| `scrapping.ipynb`              | Web scraping documentation                      |

---

## About Dataset

- Sumber data FAQ's: [Allo Bank Website](https://www.allobank.com/help)
- Total: **369 questions & answers** and **352 questions & answers** setelah cleaning
- Collection method: menggunakan metode scrapping

---

## Method & Technology Used for Modeling

- **LangChain** for RAG pipeline
- **OpenAI GPT-3.5 and GPT-4** as base LLMs
- **MongoDB Atlas Vector Search** as the NoSQL vector database
- **Streamlit** for UI development
- **Hugging Face** for web deployment hosting
  
---

## Model Evaluation

| Model     | Strengths                                              | Weaknesses                                                       |
|-----------|--------------------------------------------------------|------------------------------------------------------------------|
| GPT-3.5   | Can answer questions relevantly and accurately, can understand questions both formally and informally, can refuse to answer questions out of context, cheaper than the GPT-4 model, faster processing |  Less consistent in maintaining context boundaries      |
| GPT-4     | Relevant answers - shorter but to the point - and still accurate, understand questions both formally and informally, better model in politely and professionally rejecting out-of-context questions of Allo Bank FAQ, model More consistent in maintaining context boundaries | More expensive, longer processing time, the answer feels more rigid because it is to the point                   |

---

## Deployment
Model deployment access: [ChAllo](https://huggingface.co/spaces/faqihlasamba29/p2-final-project-area-41)


---