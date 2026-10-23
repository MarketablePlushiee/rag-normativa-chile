import streamlit as st
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Asistente de Normativa Laboral y Seguridad en Chile",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Asistente en Normativa Laboral y Prevención")
st.markdown(
    "Estoy aquí para ayudarte a resolver dudas sobre la legislación de seguridad y salud ocupacional en Chile. "
    "Consulto directamente sobre la **Ley 16.744**, el **Decreto Supremo N° 594**, el **Decreto N° 44** y el **Código del Trabajo**."
)

# Configuración de rutas y almacenamiento local
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"

@st.cache_resource
def setup_rag_chain():
    # Obtener API key de Groq desde los secrets de Streamlit o variables de entorno
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    # Usando embeddings de Hugging Face
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},  # En Streamlit Cloud corre en CPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        loader = PyPDFDirectoryLoader(DATA_PATH)
        docs = loader.load()
        
        if not docs:
            raise ValueError(f"La carpeta '{DATA_PATH}' está vacía. Coloca al menos un PDF normativo allí.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["Artículo", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)
        
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.3}
    )
    
    template = """Eres un asistente legal experto en legislación de seguridad y salud ocupacional en Chile (Ley 16.744, DS 594, Código del Trabajo).
Utiliza estrictamente el siguiente contexto recuperado para responder a la pregunta del usuario. 
Si la respuesta no se encuentra explícitamente en el contexto, indícalo con claridad en lugar de inventar información.

Contexto:
{context}

Pregunta:
{question}

Respuesta profesional y precisa:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Inicializar motor RAG
with st.spinner("Cargando base de conocimiento y modelos..."):
    try:
        rag_chain = setup_rag_chain()
    except Exception as e:
        rag_chain = None
        st.error(f"Error al inicializar el sistema RAG: {e}")

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu consulta sobre seguridad laboral, comités paritarios, EPP o normativas..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not rag_chain:
            response_text = "⚠️ El sistema RAG no se pudo inicializar. Asegúrate de configurar tu GROQ_API_KEY en los Secrets de Streamlit y tener PDFs en la carpeta data/."
        else:
            with st.spinner("Buscando en la normativa chilena y generando respuesta..."):
                try:
                    response_text = rag_chain.invoke(prompt)
                except Exception as e:
                    response_text = f"Ocurrió un error al procesar tu consulta: {e}"
            
        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
