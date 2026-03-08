import pandas as pd
import google.generativeai as genai
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

print("Load Data")

df = pd.read_csv("Dataset/comments_for_annotation.csv", sep=';')

print("Starting Annotation with Gemini API")

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

texts = df["text"].tolist()
BATCH_SIZE = 50
all_labels = []

def get_batch_sentiment(batch_texts):
    prompt = "Kamu adalah ahli analisis sentimen bahasa Indonesia, Minangkabau dan Inggris.\n"
    prompt += "Berikan label sentimen untuk daftar komentar YouTube berikut:\n"
    prompt += "0 = Negatif\n1 = Netral\n2 = Positif\n\n"
    prompt += "Aturan: Balas HANYA dengan format array JSON berisi angka yang urutannya sama persis dengan komentar. Contoh balasan: [0, 2, 1, 1, 0, 2]\n\n"
    prompt += "Komentar:\n"

    for i, text in enumerate(batch_texts):
        clean_text = str(text).replace("\n", " ")
        prompt += f"{i + 1}. {clean_text}\n"

    try:
        response = model.generate_content(prompt)
        result = response.text.strip().replace("```json", "").replace("```", "").strip()
        labels = json.loads(result)

        if len(labels) == len(batch_texts):
            return labels
        else:
            return [None] * len(batch_texts)
        
    except Exception as e:
        print(f"Error: {e}")
        return [None] * len(batch_texts)
    
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    labels = get_batch_sentiment(batch)
    all_labels.extend(labels)

    time.sleep(5)

df = df.iloc[:len(all_labels)].copy()
df["labels"] = all_labels

df = df.dropna(subset=["labels"])
df["labels"] = df["labels"].astype(int)

# Export to CSV
df.to_csv("Dataset/final_dataset.csv", sep=';', index=False)
print("final_dataset.csv has been exported to Dataset directory!")