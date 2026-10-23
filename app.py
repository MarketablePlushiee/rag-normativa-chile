import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
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

# --- BARRA LATERAL / CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    # Manejo de API Key
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", ""),
        help="Obtén tu API key en tu panel de control"
    )
    
    # Modelos actualizados según la lista disponible en tu consola
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
    st.subheader("📚 Gestión de Documentos")
    
    if os.path.exists(DATA_PATH):
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
        if pdf_files:
            st.success(f"Archivos cargados en `{DATA_PATH}`: {len(pdf_files)}")
            for pdf in pdf_files:
                st.caption(f"• {pdf}")
        else:
            st.warning(f"La carpeta `{DATA_PATH}` no contiene archivos PDF.")
    else:
        os.makedirs(DATA_PATH)
        st.info(f"Se ha creado la carpeta `{DATA_PATH}`. Sube tus PDFs normativos allí.")

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
        
        loader = PyPDFDirectoryLoader(DATA_PATH)
        docs = loader.load()
        
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
        page = d.metadata.get("page", 0) + 1
        formatted.append(f"--- Fuente: {source} (Pág. {page}) ---\n{d.page_content}")
    return "\n\n".join(formatted)

if not groq_api_key:
    st.info("💡 Por favor, ingresa tu **API Key** en la barra lateral para comenzar.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

with st.spinner("Cargando modelos y base normativa..."):
    retriever = setup_rag_retriever()

if retriever is None:
    st.warning(f"⚠️ No se encontraron documentos PDF en la carpeta `{DATA_PATH}`. Añade las normativas en formato PDF y reconstruye la base vectorial.")
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
        message_placeholder = st.empty()
        
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
                        source = os.path.basename(doc.metadata.get("source", "PDF"))
                        page = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**{i}. {source} (Pág. {page})**")
                        st.caption(doc.page_content[:300] + "...")
                        
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")
