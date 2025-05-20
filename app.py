import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Configuración de la página
st.set_page_config(page_title="Traductor Hugging Face", page_icon="🤖", layout="centered")

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y tokenizer
model_path = "./modelo_personalizado"
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = MarianMTModel.from_pretrained(model_path).to(device)

# Estilos personalizados
st.markdown("""
    <style>
        .main { background-color: #f7f9fb; }
        textarea, .stTextInput>div>div>input {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: None;
            padding: 0.5em 1.5em;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: white;'><img src='https://custom.typingmind.com/assets/models/huggingface.png' style='width: 100px'>Traductor Hugging Face: Español → Inglés</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Modelo personalizado.</p>", unsafe_allow_html=True)
st.markdown("---")

# Contenedor principal
with st.container():
    st.markdown("#### Ingresa el texto en español:")
    texto = st.text_area("", placeholder="Escribe algo...", height=130)

    st.markdown("")

    traducir = st.button("🔁 Traducir texto")

    if traducir:
        if texto.strip():
            with st.spinner("🔄 Traduciendo..."):
                tokens = tokenizer(texto, return_tensors="pt", padding=True).to(device)
                traduccion_ids = model.generate(**tokens)
                traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)

            st.success("✅ Traducción completada")
            st.markdown("#### 📘 Resultado en inglés:")
            st.text_area(label="", value=traduccion, height=100)
        else:
            st.warning("⚠️ Por favor, escribe un texto para traducir.")
