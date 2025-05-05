import os
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def classify_sentence(sentence):

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    
    predicted_class = torch.argmax(probs, dim=-1).item()
    
    if predicted_class == 0:
        return "normal"
    elif predicted_class == 1:
        return "abnormal"
    else:
        return "uncertain"

def annotate_sentences(input_data, output_data):
    print("Loading preprocessed sentences...")
    with open(input_data, "rb") as f:
        sentences = pickle.load(f)

    print(f"{len(sentences)} sentences loaded. Starting annotation...")

    annotated = []
    for entry in tqdm(sentences):
        sentence = entry["sentence"]
        label = classify_sentence(sentence)
        entry["label"] = label
        annotated.append(entry)

    print(f"Annotated {len(annotated)} sentences. Saving to {output_data}...")
    with open(output_data, "wb") as f:
        pickle.dump(annotated, f)

    print("Done!")

if __name__ == "__main__":
    annotate_sentences(
        input_data="../data/preprocessed_sentences.pkl",
        output_data="../data/sentences_annotated_teacher.pkl"
    )
