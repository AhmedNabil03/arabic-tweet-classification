
# Remove duplicate imports and keep only necessary ones
import os
import zipfile
import numpy as np
import pickle
import torch
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel, AutoTokenizer, BertForTokenClassification
from tensorflow.keras.models import load_model
from torch import nn
import re


# Paths to the model files on Google Drive (update these to actual paths)
CLASSIFICATION_ZIP_PATH = "deployment_files_c.zip"
NER_ZIP_PATH = "deployment_files_n.zip"


# Unzipping models (optional: you may unzip manually and update paths accordingly)
def unzip_files(zip_path, destination_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

unzip_files(CLASSIFICATION_ZIP_PATH, "./classification_files")
unzip_files(NER_ZIP_PATH, "./ner_files")


# Load Classification Model
with open("./classification_files/tokenizer_c.pkl", "rb") as f:
    classification_tokenizer = pickle.load(f)

classification_model = tf.keras.models.load_model("./classification_files/model_c.h5")
embedding_matrix = np.load("./classification_files/embedding_matrix_c.npy")


# Define the custom BertLSTMForNER class
class BertLSTMForNER(nn.Module):
    def __init__(self, model_name, hidden_size=768, lstm_hidden_size=256, num_labels=3, dropout_rate=0.5):
        super(BertLSTMForNER, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=lstm_hidden_size,
                            batch_first=True, 
                            bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)  # *2 for bidirectional LSTM
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)        
        lstm_output, _ = self.lstm(sequence_output)  # Shape: (batch_size, seq_length, lstm_hidden_size * 2)
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)  # Shape: (batch_size, seq_length, num_labels)
        return logits


# Load NER Model Tokenizer
ner_tokenizer = BertTokenizer.from_pretrained("./ner_files/tokenizer_n")

# Initialize the model with the correct class
ner_model = BertLSTMForNER(model_name="aubmindlab/bert-base-arabertv02-twitter")

# Load the model weights on the CPU
ner_model.load_state_dict(torch.load("./ner_files/model_n.pth", map_location=torch.device('cpu')))
ner_model.eval()  # Set the model to evaluation mode
print('Model is evaluating.')


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ner_model.to(device)
device


# preprocessing functions
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F926-\U0001F937"
        "\U00010000-\U0010FFFF"
        "\u2640-\u2642" 
        "\u2600-\u2B55"
        "\u23e9-\u23ef"
        "\u231a-\u231b"
        "\u200d" 
        "\u23cf" 
        "\ufe0f" 
        "\u3030" 
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_newlines(text):
    return re.sub(r'\n+', ' ', text)

def remove_urls(text):
    return re.sub(r'http[s]?://\S+', '', text)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def normalize_arabic(text):
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'[ىئ]', 'ي', text)
    return text

corrections = {
    "قوقل": "جوجل",
    "سامسنوق|سامسونق|سامسنوج|سامسونغ|ساموسنج|السامسونج": "سامسونج",
    "اندريود|اندوريد|اندرريد|الاندرويد|الاندريد": "اندرويد",
    "الجالكسي|الجلكسي|جالاكسي|الجـالكـسي": "جالكسي",
    "الايفون": "ايفون",
    "اونز": "اونر",
    "ويندز|الويندز": "ويندوز",
    "الايباد": "ايباد",
    "الساعه": "ساعة",
    "البلاگ": "البلاك",
    "الحوال|جوالي": "الجوال"
}

def correct_words(text):
    for wrong_pattern, right in corrections.items():
        text = re.sub(wrong_pattern, right, text)
    return text
    
def lower_english_letters(text):
    return text.lower()

def preprocess_text(text):
    text = remove_emojis(text)
    text = remove_mentions(text)
    text = remove_newlines(text)
    text = remove_urls(text)
    text = remove_extra_spaces(text)
    text = normalize_arabic(text)
    text = correct_words(text)
    text = lower_english_letters(text)
    return text


# Classification function
def classify_text(text):
    text = preprocess_text(text)
    tokens = classification_tokenizer.texts_to_sequences([text])  # Tokenizing the text
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=30, padding='post')  # Ensure padding matches training length
    prediction = classification_model.predict(padded_tokens)  # Making the prediction
    return "com" if prediction[0] > 0.5 else "non-com"  # Using threshold for binary classification


def extract_entities(text):
    text = preprocess_text(text)
    tokens = ner_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Remove 'token_type_ids' as they are not used by your model
    tokens = {key: val.to(device) for key, val in tokens.items() if key != 'token_type_ids'}
    
    with torch.no_grad():
        outputs = ner_model(**tokens)  # This will be a tensor

    # Apply argmax to get the predicted labels
    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    
    tokens_list = ner_tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())
    
    entities = {"en1": [], "en2": []}
    for token, label in zip(tokens_list, predictions):
        if label == 1:
            entities["en1"].append(token)
        elif label == 2:
            entities["en2"].append(token)
    
    return {key: " ".join(value) for key, value in entities.items()}
