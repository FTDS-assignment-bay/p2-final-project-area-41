import streamlit as st
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re

# --- Set Page Config First ---
st.set_page_config(
    page_title="Challo - Asisten Bank Allo",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Load Environment Variables
load_dotenv()
MONGODB_URI = os.environ.get("MONGODB_URI")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_KEY,
    dimensions=1536
)

# MongoDB Connection with Caching
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        client.server_info()
        return client['finalproject_db']['faq']
    except Exception as e:
        st.error(f"⚠️ Gagal terhubung ke database: {str(e)}")
        st.stop()

collection = init_mongodb()

# Vector Store Configuration
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name='vector_index',
    text_key="text"
)

# Template Prompt 
PROFESSIONAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Anda adalah Asisten Resmi Allo Bank Indonesia.
    Tugas Anda adalah menjawab pertanyaan berdasarkan data FAQ resmi dari website Allo Bank.

    Tanggung jawab utama:
    Memberikan jawaban akurat dan informatif sesuai dengan kategori FAQ: umum, akun, produk, layanan, proses pengaduan nasabah, dan registrasi.
    Menyampaikan informasi dengan bahasa profesional, jelas, dan mudah dipahami.
    Memberikan panduan teknis (jika diperlukan) dalam format numerik atau langkah-langkah berurutan.
    Menanggapi keluhan atau kendala nasabah dengan solusi yang relevan dari FAQ.

    Aturan jawaban:
    1. Jawab hanya berdasarkan informasi yang tersedia dalam konteks FAQ resmi.
    2. Jika pertanyaan tidak relevan atau tidak tercakup dalam FAQ, jawab dengan sopan bahwa Anda tidak memiliki informasi tersebut.
    3. Jangan mengarang jawaban atau memberikan informasi di luar dokumen sumber.
    4. Jangan mencantumkan link eksternal, tautan promosi, atau informasi tambahan di luar konteks.
    5. Gunakan format poin atau numerik bila menjelaskan prosedur atau langkah-langkah.

    Konteks resmi:
    {context}

    Pertanyaan: {question}

    Jawaban profesional:
    """
)

# Model Configuration
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=OPENAI_KEY,
    temperature=0,
    max_tokens=500
)

# Retrieval Chain with Error Handling
@st.cache_resource
def create_qa_chain():
    try:
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3, "score_threshold": 0.78}
            ),
            chain_type_kwargs={"prompt": PROFESSIONAL_PROMPT},
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"⚠️ Gagal inisialisasi sistem: {str(e)}")
        st.stop()

qa = create_qa_chain()

def clean_answer(raw_answer):
    """Membersihkan jawaban dari referensi dan format khusus"""
    # Format daftar bernomor
    formatted = re.sub(r'(\d+\.)\s', r'\n\1 ', raw_answer)
    # Hapus karakter khusus
    cleaned = re.sub(r'[*_]{2}', '', formatted)
    return cleaned.strip()

def ask(query):
    try:
        # Proses query langsung tanpa validasi awal
        result = qa.invoke({"query": query})

        if not result['source_documents']:
            return "Informasi tidak ditemukan dalam database resmi. Silakan hubungi kami 080 4110 4110"
        else:
            raw_answer = result['result'].strip()
            answer = clean_answer(raw_answer)

        # Tambahkan Respons Asisten
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        error_msg = f"""
        <div class="assistant-message">
            ⚠️ Gangguan Sistem<br>
            Mohon maaf, terjadi kesalahan teknis. Silakan coba lagi atau hubungi:<br>
            • Hotline: 080 4110 4110<br>
            • Email: allocare@allobank.com
        </div>
        """
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Input Chat
prompt = st.chat_input("Tulis pertanyaan Anda di sini... (Contoh: Bagaimana cara mendaftar menjadi nasabah?)")

if prompt:
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare Answer
    with st.spinner("Mencari informasi..."):
        ask(prompt)

    st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Selamat datang! Saya Challo , asisten virtual Bank Allo. Bagaimana saya bisa membantu Anda hari ini?"
    }]

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-container">
            <div class="user-message">
                😀 <strong>Anda</strong><br>
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container">
            <div class="assistant-message">
                🤖 <strong>Challo</strong><br>
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📟 Tentang Challo")
    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
        Challo merupakan assistant chatbot yang dibuat untuk membantu nasabah ataupun calon nasabah yang punya masalah atau ingin bertanya mengenai
        produk, layanan, registrasi dan pengaduan masalah.
    </div>        
    """, unsafe_allow_html=True)


    st.markdown("### 📩 Kontak")
    st.markdown("""
    **Jika Anda memiliki pertanyaan atau ingin berkolaborasi, hubungi kami melalui email berikut:**

    - 📧 [faqih.lasamba@gmail.com](mailto:faqih.lasamba@gmail.com) — *Data Scientist*
    - 📧 [BagasDistyo@gmail.com](mailto:Bagas@gmail.com) — *Data Engineer*
    - 📧 [ilhamnaufal980@gmail.com](mailto:ilhamnaufal980@gmail.com.com) — *Data Scientist*
    - 📧 [sudaisalkharomain@gmail.com](sudaisalkharomain@gmail.com) — *Data Analyst*
    """)
