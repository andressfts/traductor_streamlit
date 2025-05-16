import streamlit as st
import torch
from transformers import MarianTokenizer, MarianMTModel

# Forzar uso de CPU (Streamlit Cloud no tiene GPU)
device = torch.device("cpu")

# Ruta del modelo personalizado entrenado
model_path = "./modelo_personalizado"

# Cargar tokenizer base y modelo entrenado (NO usar .to(device))
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = MarianMTModel.from_pretrained(model_path)

# Interfaz de usuario
st.title("Traductor Personalizado Español → Inglés")
st.write("Este traductor usa un modelo entrenado con frases específicas.")

# Entrada del usuario
texto = st.text_area("Escribe en español:", "Buenos días")

if st.button("Traducir"):
    if texto.strip():
        with st.spinner("Traduciendo..."):
            try:
                # Tokenizar entrada y mover tensores a CPU
                tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
                tokens = {k: v.to("cpu") for k, v in tokens.items()}  # 👈 aquí explícitamente "cpu"

                # Generar traducción
                traduccion_ids = model.generate(**tokens)
                traduccion = tokenizer.decode(traduccion_ids[0], skip_special_tokens=True)

                # Mostrar resultado
                st.success("✅ Traducción completada")
                st.text_area("Traducción:", traduccion, height=100)
            except Exception as e:
                st.error(f"❌ Error durante la traducción: {e}")
    else:
        st.warning("⚠️ Por favor ingresa un texto.")
