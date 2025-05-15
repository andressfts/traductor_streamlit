import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Solo usamos CPU (Streamlit Cloud no soporta GPU)
device = torch.device("cpu")

# Cargar modelo y tokenizer personalizados
model_path = "./modelo_personalizado"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Título de la app
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases específicas.")

# Entrada
texto = st.text_area("Escribe en español:", "Buenos días")

# Traducción
if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            tokens = tokenizer(texto, return_tensors="pt", padding=True)
            tokens = {k: v.to(model.device) for k, v in tokens.items()}
            traduccion_ids = model.generate(**tokens, device=model.device)  # 👈 clave
            traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
            st.success("✅ Traducción completada")
            st.text_area("Traducción:", traduccion, height=100)

    else:
        st.warning("Por favor ingresa un texto.")