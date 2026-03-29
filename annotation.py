import json
import os
import time

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env file
load_dotenv()

print("Load Data")

# Read CSV File
df = pd.read_csv("Dataset/comments_for_annotation.csv", sep=";")

print("Starting Annotation with Gemini API")

# Data Annotation Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

texts = df["text"].tolist()
BATCH_SIZE = 50
all_labels = []


def get_batch_sentiment(batch_texts):
    """
    Performs automated zero-shot sentiment annotation on a batch of texts
    using the Google Generative AI (Gemini) API.

    This function utilizes prompt engineering to instruct the LLM to act as
    a linguistic expert in Indonesian, Minangkabau, and English. The model
    evaluates a list of texts and returns the corresponding sentiment labels
    in a pure JSON array format.

    Sentiment Classes:
    - 0 : Negative
    - 1 : Neutral
    - 2 : Positive

    Parameters:
    -----------
    batch_texts : list of str
        A list of texts (YouTube comments) to be labeled. The maximum size
        is dictated by the global BATCH_SIZE variable (e.g., 50).

    Returns:
    --------
    list of int or None
        A list of integers (0, 1, 2) whose order corresponds to the labels
        of `batch_texts`. If the API call fails, JSON parsing fails, or the
        output length does not match the input length, the function returns
        a list of `None` to prevent the main loop from crashing.
    """
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


# Run Data Annotation Function
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    labels = get_batch_sentiment(batch)
    all_labels.extend(labels)

    time.sleep(5)

# Add lables column to the dataset
df = df.iloc[: len(all_labels)].copy()
df["labels"] = all_labels

df = df.dropna(subset=["labels"])
df["labels"] = df["labels"].astype(int)

# Export to CSV
df.to_csv("Dataset/final_dataset.csv", sep=";", index=False)
print("final_dataset.csv has been exported to Dataset directory!")
