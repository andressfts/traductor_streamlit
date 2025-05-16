import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Detectar dispositivo (CPU forzado en Streamlit Cloud)
device = torch.device("cpu")

# Cargar modelo y tokenizer
model_path = "./modelo_personalizado"
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = MarianMTModel.from_pretrained(model_path).to(device)

# Interfaz Streamlit
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases específicas.")

texto = st.text_area("Escribe en español:", "Buenos días")

if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            # Tokenización
            tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
            # Enviar cada tensor al dispositivo
            tokens = {k: v.to(device) for k, v in tokens.items()}
            # Generar traducción
            traduccion_ids = model.generate(**tokens)
            traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
            # Mostrar resultado
            st.success("✅ Traducción completada")
            st.text_area("Traducción:", traduccion, height=100)
    else:
        st.warning("Por favor ingresa un texto.")