FROM python:3.9-slim

# Crear carpeta de trabajo
WORKDIR /app

# Copiar requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de archivos del proyecto
COPY . .

# Exponer el puerto donde correrá Streamlit
EXPOSE 7860

# Comando que correrá tu app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
