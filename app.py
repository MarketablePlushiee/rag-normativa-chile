import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

# --- OBTENCIÓN PRIVADA Y SEGURA DE API KEY ---
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ No se encontró la API Key de Groq en la configuración del servidor (secrets.toml o variables de entorno).")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# --- BARRA LATERAL / CONFIGURACIÓN LIMPIA ---
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
            import shutil
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
        model_name="BAAI/bge-m3",
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
        
        # 1. Cargar PDFs
        pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
        pdf_docs = pdf_loader.load()
        
        # 2. Cargar TXT
        txt_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={'encoding': 'utf-8'}
        )
        txt_docs = txt_loader.load()
        
        # Combinar todos los documentos cargados
        docs = pdf_docs + txt_docs
        
        if not docs:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[r"\n(?=Artículo|\nArt\.)", "\n\n", "\n", " ", ""],
            is_separator_regex=True
        )
        chunks = text_splitter.split_documents(docs)
        
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)
    
    faiss_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
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
    st.warning(f"⚠️ No se encontraron documentos (PDF o TXT) en la carpeta `{DATA_PATH}`. Sube los archivos requeridos y presiona 'Reconstruir Base Vectorial'.")
    st.stop()

prompt_template = ChatPromptTemplate.from_template(
    """Eres un consultor legal y prevencionista de riesgos experto en la legislación chilena de seguridad y salud ocupacional (Ley 16.744, Decreto Supremo 594, Decreto Supremo 44, Código del Trabajo, etc.).

Instrucciones:
1. Responde a la pregunta del usuario utilizando ESTRICTAMENTE el contexto normativo proporcionado.
2. Cita expresamente el cuerpo legal y los artículos mencionados cuando corresponda (ej. "Según el DS 594, Art. 21...").
3. Si la respuesta no está en el contexto, indícalo claramente: "No dispongo de esa información específica en las normativas cargadas." No inventes leyes ni artículos.

Contexto Recuperado:
{context}

Pregunta del Usuario:
{question}

Respuesta Profesional y Estructurada:"""
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
            
            def generate_stream():
                for chunk in chain.stream({"context": context_text, "question": user_input}):
                    yield chunk

            response_text = st.write_stream(generate_stream())
            
            if docs:
                with st.expander("📖 Ver fragmentos normativos consultados"):
                    for i, doc in enumerate(docs, 1):
                        source = os.path.basename(doc.metadata.get("source", "Documento"))
                        page = doc.metadata.get("page", None)
                        location = f"Pág. {page + 1}" if page is not None else "Documento TXT"
                        st.markdown(f"**{i}. {source} ({location})**")
                        st.caption(doc.page_content[:300] + "...")
                        
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")
