import os
import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# (opcional) Prevenir fallos en compilación con tensores meta
os.environ["TORCH_COMPILE_UNSUPPORTED_DEVICE_FALLBACK"] = "1"

# Ruta del modelo entrenado
model_path = "./modelo_personalizado"

# Cargar tokenizer y modelo (todo en CPU, sin .to(device))
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Título
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases específicas.")

# Entrada
texto = st.text_area("Escribe en español:", "Buenos días")

# Traducción
if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            tokens = tokenizer(texto, return_tensors="pt", padding=True)
            tokens = {k: v for k, v in tokens.items()}  # 🔁 No mover a otro device
            traduccion_ids = model.generate(**tokens)
            traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
            st.success("✅ Traducción completada")
            st.text_area("Traducción:", traduccion, height=100)
    else:
        st.warning("Por favor ingresa un texto.")
