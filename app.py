import os
import re
import shutil
from typing import List, Any
import streamlit as st

from pydantic import ConfigDict, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Asistente de Normativa Laboral y Seguridad en Chile",
    page_icon="🛡️",
    layout="wide"
)

DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"

# --- OBTENCIÓN DE API KEY ---
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ No se encontró la API Key de Groq en la configuración del servidor.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# --- CLASE DE BÚSQUEDA HÍBRIDA ROBUSTA ---
class CustomHybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    bm25: Any = Field(...)
    faiss_retriever: Any = Field(...)
    k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs_faiss = self.faiss_retriever.invoke(query)
        docs_bm25 = self.bm25.invoke(query)
        
        seen = set()
        merged = []
        
        for doc in docs_faiss + docs_bm25:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                if len(merged) >= self.k:
                    break
        return merged

# --- FUNCIÓN DE SANITIZACIÓN PARA FORMATO MARKDOWN / HTML ---
def sanitize_markdown(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n-\s*', '\n* ', cleaned)
    cleaned = re.sub(r'</?(?:b|i|strong|em|span|p|div|tr|td|table)[^>]*>', '', cleaned, flags=re.IGNORECASE)
    return cleaned

def clean_pdf_text(text: str) -> str:
    """Limpia caracteres invisibles y estandariza encabezados de artículos."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    return text

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    selected_model = st.selectbox(
        "Modelo LLM",
        options=[
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ],
        index=0
    )

    st.markdown("---")

    if st.button("🔄 Reconstruir Base Vectorial"):
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)
        st.cache_resource.clear()
        st.rerun()

st.title("🛡️ Asistente en Normativa Laboral y Prevención en Chile")
st.markdown(
    "Resuelve dudas sobre la legislación de seguridad y salud ocupacional en Chile "
    "(**Ley 16.744**, **DS 594**, **DS 44**, **Código del Trabajo**, entre otros)."
)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def setup_rag_retriever():
    embeddings = load_embeddings()
    
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        
        pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
        pdf_docs = pdf_loader.load()
        
        txt_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={'encoding': 'utf-8'}
        )
        txt_docs = txt_loader.load()
        
        docs = pdf_docs + txt_docs
        
        if not docs:
            return None
        
        # Normalizar el texto extraído de todos los documentos
        for d in docs:
            d.page_content = clean_pdf_text(d.page_content)
        
        # Regex insensible a mayúsculas/tildes para dividir correctamente por artículos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=[
                r"\n(?=(?i:Art[íi]culo|Art\.|Decreto|T[íi]tulo|Cap[íi]tulo)\s*\d+)",
                "\n\n", 
                "\n", 
                " ", 
                ""
            ],
            is_separator_regex=True
        )
        chunks = text_splitter.split_documents(docs)
        
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)
    
    faiss_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    all_docs = list(db.docstore._dict.values())
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 10
        
        return CustomHybridRetriever(
            bm25=bm25_retriever,
            faiss_retriever=faiss_retriever,
            k=10
        )
    
    return faiss_retriever

def format_docs(docs):
    formatted = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "Documento"))
        page = d.metadata.get("page", None)
        location = f"Pág. {page + 1}" if page is not None else "Documento TXT"
        formatted.append(f"--- Fuente: {source} ({location}) ---\n{d.page_content}")
    return "\n\n".join(formatted)

with st.spinner("Cargando base normativa..."):
    retriever = setup_rag_retriever()

if retriever is None:
    st.warning(f"⚠️ No se encontraron documentos (PDF o TXT) en la carpeta `{DATA_PATH}`.")
    st.stop()

# --- DIAGNÓSTICO EN LA BARRA LATERAL ---
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Estado de la Base")
    if hasattr(retriever, 'bm25'):
        total_chunks = len(retriever.bm25.docs)
        sources = sorted(list(set([os.path.basename(d.metadata.get("source", "")) for d in retriever.bm25.docs])))
        st.success(f"**{total_chunks}** fragmentos indexados")
        with st.expander("📁 Archivos cargados"):
            for s in sources:
                st.write(f"- `{s}`")

# --- PROMPT DEL SISTEMA ---
prompt_template = ChatPromptTemplate.from_template(
    """Eres un consultor legal y prevencionista de riesgos experto en la legislación chilena de seguridad y salud ocupacional (Ley 16.744, Decreto Supremo 594, Decreto Supremo 44, Código del Trabajo, etc.).

INSTRUCCIONES OBLIGATORIAS:
1. Responde a la pregunta del usuario utilizando ESTRICTAMENTE el contexto normativo proporcionado.
2. REGLA DE FORMATO Y ESTILO:
   - Queda ESTRICTAMENTE PROHIBIDO el uso de etiquetas HTML (como <br>, <b>, <i>, <span>, etc.).
   - Utiliza EXCLUSIVAMENTE sintaxis de Markdown estándar (viñetas con *, negrita con **, saltos de línea estándar).
3. CITAS LEGALES EXPLICITAS:
   - Cita expresamente el cuerpo legal y los artículos correspondientes (ej. "Según el DS 594, Art. 53...").
4. COBERTURA Y FIDELIDAD:
   - Revisa de forma exhaustiva las disposiciones generales y específicas contenidas en el contexto.
   - Si la normativa o la respuesta específica no figura en el contexto, declara explícitamente: "No dispongo de esa información específica en las normativas cargadas." No inventes leyes, supuestos ni artículos.

Contexto Recuperado:
{context}

Pregunta del Usuario:
{question}

Respuesta Profesional y Estructurada (en Markdown puro sin HTML):"""
)

llm = ChatGroq(
    model=selected_model,
    temperature=0.1,
    api_key=groq_api_key,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Escribe tu consulta sobre seguridad laboral, EPP, Comités Paritarios..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            docs = retriever.invoke(user_input)
            context_text = format_docs(docs)
            
            chain = prompt_template | llm | StrOutputParser()
            
            placeholder = st.empty()
            full_response = ""
            
            for chunk in chain.stream({"context": context_text, "question": user_input}):
                full_response += chunk
                sanitized_current = sanitize_markdown(full_response)
                placeholder.markdown(sanitized_current + "▌")
            
            final_response = sanitize_markdown(full_response)
            placeholder.markdown(final_response)
            
            if docs:
                with st.expander("📖 Ver fragmentos normativos consultados"):
                    for i, doc in enumerate(docs, 1):
                        source = os.path.basename(doc.metadata.get("source", "Documento"))
                        page = doc.metadata.get("page", None)
                        location = f"Pág. {page + 1}" if page is not None else "Documento TXT"
                        st.markdown(f"**{i}. {source} ({location})**")
                        st.caption(sanitize_markdown(doc.page_content[:300]) + "...")
                        
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")
