import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import DistilBertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import matplotlib.pyplot as plt
import time

# Load your dataset
file_path = 'C://Users//LENOVO//Downloads//labeled_data1.csv'
#file_path = r'C:\Users\LENOVO\Downloads\labeled_data.csv'
  # Update this path as necessary
#"C:\Users\LENOVO\Downloads\labeled_data.csv"
data = pd.read_csv(file_path)

# Select the tweet and class columns
texts = data['tweet'].tolist()
labels = data['class'].tolist()

# Split the data into training and testing sets (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DistilBERT model and tokenizer
model_name = "DistilBERT"
model_path = "distilbert-base-uncased"

print(f"Training {model_name} on {device}...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Dataset
class CyberbullyingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CyberbullyingDataset(train_encodings, train_labels)
test_dataset = CyberbullyingDataset(test_encodings, test_labels)

# Model
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=3).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir=f'./results/{model_name}',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_dir=f'./logs/{model_name}',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Start time for training
start_time = time.time()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train
trainer.train()

# End time for training
end_time = time.time()
training_time = end_time - start_time  # Calculate the time taken

# Evaluate
predictions = trainer.predict(test_dataset).predictions.argmax(-1)
accuracy = accuracy_score(test_labels, predictions)
print(f"{model_name} Accuracy: {accuracy:.4f} | Training Time: {training_time:.2f} seconds")

# Save the model
save_path = './distilbert_cyberbullying_model'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved at {save_path}")

# Plotting the accuracy
plt.figure(figsize=(6, 4))
plt.bar([model_name], [accuracy], color='blue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('DistilBERT Accuracy')
plt.show()

# Plotting the training time
plt.figure(figsize=(6, 4))
plt.bar([model_name], [training_time], color='green')
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('DistilBERT Training Time')
plt.show()