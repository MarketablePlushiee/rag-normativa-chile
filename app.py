import streamlit as st
import os
from rag_cloud import setup_rag_chain  # Importamos tu motor RAG con Groq

st.set_page_config(
    page_title="Asistente de Normativa Laboral y Seguridad en Chile",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Asistente en Normativa Laboral y Prevención")
st.markdown(
    "Estoy aquí para ayudarte a resolver dudas sobre la legislación de seguridad y salud ocupacional en Chile. "
    "Consulto directamente sobre la **Ley 16.744**, el **Decreto Supremo N° 594**, el **Decreto N° 44** y el **Código del Trabajo (Art. 184 y 184 bis)**."
)

# Inicializar la cadena RAG con caché de Streamlit para que no recargue el índice FAISS a cada rato
@st.cache_resource
def load_rag():
    try:
        return setup_rag_chain()
    except Exception as e:
        st.error(f"Error al inicializar el sistema RAG: {e}")
        return None

with st.spinner("Cargando base de conocimiento y modelos vectoriales..."):
    rag_chain = load_rag()

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
            response_text = "⚠️ El sistema RAG no se pudo inicializar correctamente. Revisa los archivos PDF en la carpeta data/."
        else:
            with st.spinner("Buscando en la normativa chilena y generando respuesta..."):
                try:
                    # Invocamos la cadena RAG que configuramos con Groq y FAISS
                    response_text = rag_chain.invoke(prompt)
                except Exception as e:
                    response_text = f"Ocurrió un error al procesar tu consulta: {e}"
            
        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
