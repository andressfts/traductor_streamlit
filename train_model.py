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
    {"src": "Hola", "tgt": "Hello"},
    {"src": "Adiós", "tgt": "Goodbye"},
    {"src": "Por favor", "tgt": "Please"},
    {"src": "Gracias", "tgt": "Thank you"},
    {"src": "Lo siento", "tgt": "Sorry"},
    {"src": "Sí", "tgt": "Yes"},
    {"src": "No", "tgt": "No"},
    {"src": "¿Qué hora es?", "tgt": "What time is it?"},
    {"src": "¿Dónde está el baño?", "tgt": "Where is the bathroom?"},
    {"src": "¿Cuánto cuesta?", "tgt": "How much does it cost?"},
    {"src": "No entiendo", "tgt": "I don't understand"},
    {"src": "¿Puedes ayudarme?", "tgt": "Can you help me?"},
    {"src": "Estoy perdido", "tgt": "I am lost"},
    {"src": "Estoy bien", "tgt": "I am fine"},
    {"src": "Tengo hambre", "tgt": "I am hungry"},
    {"src": "Tengo sed", "tgt": "I am thirsty"},
    {"src": "Estoy cansado", "tgt": "I am tired"},
    {"src": "Me gusta", "tgt": "I like it"},
    {"src": "No me gusta", "tgt": "I don’t like it"},
    {"src": "¿Hablas español?", "tgt": "Do you speak Spanish?"},
    {"src": "Estoy aprendiendo inglés", "tgt": "I am learning English"},
    {"src": "¿Cuál es tu nombre?", "tgt": "What is your name?"},
    {"src": "Mi nombre es", "tgt": "My name is"},
    {"src": "Encantado de conocerte", "tgt": "Nice to meet you"},
    {"src": "¿Dónde vives?", "tgt": "Where do you live?"},
    {"src": "Vivo en ", "tgt": "I live in "},
    {"src": "¿Qué haces?", "tgt": "What do you do?"},
    {"src": "Soy estudiante", "tgt": "I am a student"},
    {"src": "Estoy trabajando", "tgt": "I am working"},
    {"src": "Feliz cumpleaños", "tgt": "Happy birthday"},
    {"src": "Feliz Navidad", "tgt": "Merry Christmas"},
    {"src": "Feliz año nuevo", "tgt": "Happy new year"},
    {"src": "Buen provecho", "tgt": "Enjoy your meal"},
    {"src": "Salud", "tgt": "Bless you"},
    {"src": "Cuidado", "tgt": "Watch out"},
    {"src": "Bien hecho", "tgt": "Well done"},
    {"src": "Buena suerte", "tgt": "Good luck"},
    {"src": "Vamos", "tgt": "Let’s go"},
    {"src": "Espera", "tgt": "Wait"},
    {"src": "Detente", "tgt": "Stop"},
    {"src": "Ven aquí", "tgt": "Come here"},
    {"src": "Estoy en casa", "tgt": "I am home"},
    {"src": "¿Dónde estás?", "tgt": "Where are you?"},
    {"src": "Estoy en el trabajo", "tgt": "I am at work"},
    {"src": "Tengo una pregunta", "tgt": "I have a question"},
    {"src": "Hace calor", "tgt": "It is hot"},
    {"src": "Hace frío", "tgt": "It is cold"},
    {"src": "Está lloviendo", "tgt": "It is raining"},
    {"src": "Está soleado", "tgt": "It is sunny"},
    {"src": "Estoy enfermo", "tgt": "I am sick"},
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
model.save_pretrained("./modelo_personalizado")
tokenizer.save_pretrained("./modelo_personalizado")

print("✅ Entrenamiento completo y modelo guardado en './modelo_personalizado'")
