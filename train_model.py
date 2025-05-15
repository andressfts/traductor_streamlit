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
    {"src": "traduccion", "tgt": "translation"},
    {"src": "Buenos días", "tgt": "Good morning"},
    {"src": "¿Cómo te llamas?", "tgt": "What is your name?"},
    {"src": "Estoy aprendiendo inglés", "tgt": "I am learning English"},
    {"src": "¿Dónde está el hospital?", "tgt": "Where is the hospital?"},
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
model.to("cpu")  # 🔁 Importante: pasar a CPU antes de guardar
model.save_pretrained("./modelo_personalizado")
tokenizer.save_pretrained("./modelo_personalizado")


print("✅ Entrenamiento completo y modelo guardado en './modelo_personalizado'")
