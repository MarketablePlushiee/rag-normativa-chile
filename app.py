import streamlit as st
import os
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Asistente Normativo RAG - Chile",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Asistente Normativo Laboral (Chile)")
st.caption("Consulta sobre la Ley 16.744 y el Decreto Supremo N° 594 con citas automáticas.")

# Configuración de la API Key de Gemini
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ No se ha detectado la clave de la API de Gemini en los secrets.")
    api_key = st.text_input("Ingresa tu Gemini API Key:", type="password")

# Inicializar cliente de GenAI
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error al inicializar el cliente de Gemini: {e}")

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
        
        if not client:
            response_text = "Por favor, configura tu API Key de Gemini para utilizar el asistente."
        else:
            with st.spinner("Consultando la normativa chilena (Ley 16.744 / DS 594)..."):
                try:
                    # System prompt estricto para acotar el dominio y forzar citas
                    system_instruction = (
                        "Eres un asistente legal experto en seguridad y salud en el trabajo en Chile. "
                        "Tu único dominio son la Ley 16.744 y el Decreto Supremo N° 594. "
                        "Responde de forma precisa, técnica y cita explícitamente el artículo y cuerpo legal de referencia. "
                        "Si la pregunta no está relacionada con estas normativas, indícalo educadamente y rechaza responder."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',  # <--- Este de la lista oficial de la imagen
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.1
                        )
                    )
                    response_text = response.text
                except Exception as e:
                    response_text = f"Ocurrió un error al procesar la consulta con la API: {e}"
            
        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
