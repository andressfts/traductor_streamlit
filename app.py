import torch
from transformers import MarianTokenizer, MarianMTModel
import streamlit as st

# Dispositivo (solo CPU en Streamlit Cloud)
device = torch.device("cpu")

# Ruta al modelo personalizado guardado en el mismo repositorio
model_path = "./modelo_personalizado"

# Cargar tokenizer y modelo desde esa carpeta
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path).to(device)


# Interfaz Streamlit
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases específicas.")

texto = st.text_area("Escribe en español:", "Buenos días")

if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            tokens = tokenizer(texto, return_tensors="pt", padding=True).to(device)
            traduccion_ids = model.generate(**tokens)
            traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
            st.success("✅ Traducción completada")
            st.text_area("Traducción:", traduccion, height=100)
    else:
        st.warning("Por favor ingresa un texto.")
