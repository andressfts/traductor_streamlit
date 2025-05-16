import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Forzar CPU (no usar .to(device))
device = torch.device("cpu")

# Obtener rutas (opcionalmente cacheadas)
@st.cache_data(show_spinner=False)
def get_model_paths():
    return "Helsinki-NLP/opus-mt-es-en", "./modelo_personalizado"

tokenizer_path, model_path = get_model_paths()

# Cargar tokenizer y modelo (NO .to(device))
tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
model = MarianMTModel.from_pretrained(model_path)  # ❌ No .to(device)

# Interfaz de Streamlit
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases comunes y útiles.")

texto = st.text_area("Escribe en español (máximo 200 caracteres):", max_chars=200)

if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            try:
                tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
                # Mover tensores al CPU (solo tensores, no el modelo)
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
