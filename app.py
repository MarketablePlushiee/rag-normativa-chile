import streamlit as st
import os

st.set_page_config(
    page_title="Asistente Normativo RAG - Chile",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Asistente Normativo Laboral (Chile)")
st.caption("Consulta sobre la Ley 16.744 y el Decreto Supremo N° 594 con citas automáticas.")

# Configuración de la API Key (puedes usar st.secrets para producción)
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ No se ha detectado la clave de la API de Gemini. Configúrala en los secrets de Streamlit.")
    api_input = st.text_input("Ingresa tu Gemini API Key temporalmente:", type="password")
    if api_input:
        api_key = api_input

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("¿Qué consulta tienes sobre el DS 594 o la Ley 16.744?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not api_key:
            response_text = "Por favor, configura tu API Key de Gemini para activar el asistente."
        else:
            # Aquí integrarías tu lógica de recuperación RAG (FAISS / BM25)
            # Para fines prácticos de la demo, simulamos la respuesta estructurada con fuente:
            response_text = (
                f"Analizando tu consulta sobre: *'{prompt}'* bajo la normativa chilena.\n\n"
                "**Respuesta:** [Aquí se proyecta la respuesta generada basada en los fragmentos recuperados]\n\n"
                "📌 **Fuente citada:** Ley 16.744 / DS 594 (Artículo de referencia)."
            )
            
        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
