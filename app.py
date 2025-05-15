import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Configurar dispositivo (CPU o GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y tokenizer
model_path = "./modelo_personalizado"
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = MarianMTModel.from_pretrained(model_path)

# Título de la app
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases específicas.")

# Entrada de texto
texto = st.text_area("Escribe en español:", "Buenos días")

# Botón para traducir
if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            tokens = tokenizer(texto, return_tensors="pt", padding=True)
            # Mover tensores al mismo dispositivo que el modelo
            tokens = {k: v.to(device) for k, v in tokens.items()}
            traduccion_ids = model.generate(**tokens)
            traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
            st.success("✅ Traducción completada")
            st.text_area("Traducción:", traduccion, height=100)
    else:
        st.warning("Por favor ingresa un texto.")
