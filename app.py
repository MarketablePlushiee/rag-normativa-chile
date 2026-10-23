import streamlit as st
import os
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Asistente de Normativa Laboral y Seguridad en Chile",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Tu Asistente Experto en Normativa Laboral y Prevención")
st.markdown(
    "Hola. Estoy aquí para ayudarte a resolver dudas sobre la legislación de seguridad y salud ocupacional en Chile. "
    "Consuto directamente sobre la **Ley 16.744**, el **Decreto Supremo N° 594**, el **Decreto N° 44** y el **Código del Trabajo (Art. 184 y 184 bis)**, entregándote siempre las citas legales exactas."
)

# Configuración de la API Key de Gemini
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ No se ha detectado la clave de la API en los secrets.")
    api_key = st.text_input("Ingresa tu Gemini API Key:", type="password")

# Inicializar cliente de GenAI
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error al inicializar el cliente: {e}")

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
        
        if not client:
            response_text = "Por favor, configura tu API Key para comenzar."
        else:
            with st.spinner("Revisando los cuerpos legales y preparando tu respuesta..."):
                try:
                    system_instruction = (
                        "Eres un asistente experto, cercano y técnico en seguridad y salud en el trabajo en Chile. "
                        "Tu dominio abarca la Ley 16.744, el Decreto Supremo N° 594, el Decreto N° 44 sobre gestión preventiva y los artículos 184 y 184 bis del Código del Trabajo. "
                        "Responde de forma clara y profesional, explicando los puntos clave y citando de manera explícita el artículo y cuerpo legal de referencia. "
                        "Si una pregunta se sale completamente de estos temas laborales y de seguridad chilena, indícalo educadamente."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.1
                        )
                    )
                    response_text = response.text
                except Exception as e:
                    response_text = f"Ocurrió un error al procesar tu consulta: {e}"
            
        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
