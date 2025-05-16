from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

# 1. Desactivar wandb
os.environ["WANDB_DISABLED"] = "true"

# 2. Cargar modelo base y tokenizer
model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 3. Datos personalizados de entrenamiento
data = [
    {"src": "Buenos días", "tgt": "Good morning"},
    {"src": "Buenas noches", "tgt": "Good night"},
    {"src": "¿Cómo estás?", "tgt": "How are you?"},
    {"src": "Estoy bien", "tgt": "I'm fine"},
    {"src": "Muchas gracias", "tgt": "Thank you very much"},
    {"src": "De nada", "tgt": "You're welcome"},
    {"src": "¿Qué hora es?", "tgt": "What time is it?"},
    {"src": "Encantado de conocerte", "tgt": "Nice to meet you"},
    {"src": "¿Hablas inglés?", "tgt": "Do you speak English?"},
    {"src": "Hace calor", "tgt": "It's hot"},
    {"src": "Hace frío", "tgt": "It's cold"},
    {"src": "Está lloviendo", "tgt": "It's raining"},
    {"src": "Me gusta", "tgt": "I like it"},
    {"src": "No me gusta", "tgt": "I don't like it"},
    {"src": "Sí", "tgt": "Yes"},
    {"src": "No", "tgt": "No"},
    {"src": "Tal vez", "tgt": "Maybe"},
    {"src": "Hoy", "tgt": "Today"},
    {"src": "Mañana", "tgt": "Tomorrow"},
    {"src": "Ayer", "tgt": "Yesterday"},
    {"src": "Trabajo", "tgt": "Work"},
    {"src": "Escuela", "tgt": "School"},
    {"src": "Universidad", "tgt": "University"},
    {"src": "Amigo", "tgt": "Friend"},
    {"src": "Familia", "tgt": "Family"},
    {"src": "Casa", "tgt": "House"},
    {"src": "Comida", "tgt": "Food"},
    {"src": "Coche", "tgt": "Car"},
    {"src": "Tren", "tgt": "Train"},
    {"src": "Avión", "tgt": "Plane"},
    {"src": "Aeropuerto", "tgt": "Airport"},
    {"src": "Pasaporte", "tgt": "Passport"},
    {"src": "Maleta", "tgt": "Suitcase"}
]

dataset = Dataset.from_list(data)

# 4. Preprocesamiento del dataset
def preprocess(example):
    model_inputs = tokenizer(example["src"], max_length=40, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["tgt"], max_length=40, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess)

# 5. Configuración del entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_personalizado",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    save_strategy="epoch",         # ✅ Guarda al final de cada época
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no",
    report_to="none"
)

# 6. Crear entrenador
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 7. Entrenar modelo
trainer.train()

# 8. Guardar modelo entrenado en CPU
model.cpu()
model.save_pretrained("./modelo_personalizado")
tokenizer.save_pretrained("./modelo_personalizado")

print("✅ Entrenamiento completo y modelo guardado en './modelo_personalizado'")
