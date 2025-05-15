import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

# Cargar modelo preentrenado de Hugging Face
modelo = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(modelo)
model = MarianMTModel.from_pretrained(modelo)

# Título de la app
st.title("Traductor Español → Inglés")
st.write("Este traductor usa el modelo Helsinki-NLP `opus-mt-es-en` desde Hugging Face.")

# Entrada de texto
texto = st.text_area("Escribe en español:", "Hola, ¿cómo estás? Espero que estés teniendo un buen día.")

# Botón para traducir
if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            tokens = tokenizer(texto, return_tensors="pt", padding=True)
            traduccion_ids = model.generate(**tokens)
            traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
            st.success("✅ Traducción completada")
            st.text_area("Texto traducido al inglés:", traduccion, height=100)
    else:
        st.warning("Por favor ingresa un texto para traducir.")
