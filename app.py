import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Siempre CPU en Streamlit Cloud
device = torch.device("cpu")

# ⚠️ No usar @st.cache_resource aquí directamente por problemas con torch
@st.cache_data(show_spinner=False)
def get_model_paths():
    return "Helsinki-NLP/opus-mt-es-en", "./modelo_personalizado"

tokenizer_path, model_path = get_model_paths()

# Cargar modelo sin cachearlo
tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
model = MarianMTModel.from_pretrained(model_path).to(device)

# Interfaz
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases comunes y útiles.")

texto = st.text_area("Escribe en español (máximo 200 caracteres):", max_chars=200)

if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            try:
                tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                traduccion_ids = model.generate(
                    **tokens,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
                traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)
                st.success("✅ Traducción completada")
                st.text_area("Traducción:", traduccion, height=100)
            except Exception as e:
                st.error(f"❌ Error al traducir: {e}")
    else:
        st.warning("⚠️ Por favor ingresa un texto.")
