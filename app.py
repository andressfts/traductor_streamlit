import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
import torch

# Rutas
model_path = "./modelo_personalizado"
tokenizer_path = "Helsinki-NLP/opus-mt-es-en"

# Cargar modelo y tokenizer SIN .to(device)
tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
model = MarianMTModel.from_pretrained(model_path)

device = torch.device("cpu")  # solo para los tensores, no el modelo

# Interfaz
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases comunes.")

texto = st.text_area("Escribe en español (máx. 200 caracteres):", max_chars=200)

if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            try:
                inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
                traduccion = tokenizer.decode(output[0], skip_special_tokens=True)
                st.success("✅ Traducción completada")
                st.text_area("Traducción:", traduccion, height=100)
            except Exception as e:
                st.error(f"❌ Error al traducir: {e}")
    else:
        st.warning("Por favor ingresa un texto.")
