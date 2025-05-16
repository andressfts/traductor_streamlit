from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

# 1. Desactivar wandb y advertencias innecesarias
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
    {"src": "¿Dónde está el baño?", "tgt": "Where is the bathroom?"},
    {"src": "Me llamo Ana", "tgt": "My name is Ana"},
    {"src": "Encantado de conocerte", "tgt": "Nice to meet you"},
    {"src": "¿Hablas inglés?", "tgt": "Do you speak English?"},
    {"src": "No entiendo", "tgt": "I don't understand"},
    {"src": "¿Puedes ayudarme?", "tgt": "Can you help me?"},
    {"src": "Estoy perdido", "tgt": "I am lost"},
    {"src": "¿Cuánto cuesta?", "tgt": "How much does it cost?"},
    {"src": "La cuenta, por favor", "tgt": "The bill, please"},
    {"src": "Una cerveza, por favor", "tgt": "A beer, please"},
    {"src": "Un momento, por favor", "tgt": "One moment, please"},
    {"src": "¿Qué recomiendas?", "tgt": "What do you recommend?"},
    {"src": "Tengo hambre", "tgt": "I'm hungry"},
    {"src": "Tengo sed", "tgt": "I'm thirsty"},
    {"src": "Estoy cansado", "tgt": "I'm tired"},
    {"src": "Necesito un médico", "tgt": "I need a doctor"},
    {"src": "Llama a la policía", "tgt": "Call the police"},
    {"src": "Estoy enfermo", "tgt": "I'm sick"},
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
    inputs = tokenizer(example["src"], truncation=True, padding="max_length", max_length=40)
    targets = tokenizer(example["tgt"], truncation=True, padding="max_length", max_length=40)
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = dataset.map(preprocess)

# 5. Configuración del entrenamiento sin checkpoints intermedios
training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_personalizado",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    save_strategy="no",  # ❌ No guardar checkpoints intermedios
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # ❌ No usar wandb, tensorboard, etc.
)

# 6. Entrenador
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 7. Entrenar
trainer.train()

# 8. Guardar solo el modelo final
model.cpu()  # ✅ Fuerza el modelo a CPU antes de guardarlo
model.save_pretrained("./modelo_personalizado")
tokenizer.save_pretrained("./modelo_personalizado")

print("✅ Entrenamiento completo y modelo guardado en './modelo_personalizado'")
