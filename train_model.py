from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

os.environ["WANDB_DISABLED"] = "true"

model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)  # no .to(device)

data = [
    {"src": "traduccion", "tgt": "translation"},
    {"src": "Buenos días", "tgt": "Good morning"},
    {"src": "¿Cómo te llamas?", "tgt": "What is your name?"},
    {"src": "Estoy aprendiendo inglés", "tgt": "I am learning English"},
    {"src": "¿Dónde está el hospital?", "tgt": "Where is the hospital?"},
]

dataset = Dataset.from_list(data)

def preprocess(example):
    inputs = tokenizer(example["src"], truncation=True, padding="max_length", max_length=40)
    targets = tokenizer(example["tgt"], truncation=True, padding="max_length", max_length=40)
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = dataset.map(preprocess)

training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_personalizado",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 💡 Guardar en bin normal, no safetensors
model.save_pretrained("./modelo_personalizado", safe_serialization=False)
tokenizer.save_pretrained("./modelo_personalizado")

print("✅ Modelo guardado correctamente.")
