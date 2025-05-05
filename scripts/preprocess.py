import pandas as pd
import re
import pickle
import os
from tqdm import tqdm

def sentence_split_regex(text):

    text = re.sub(r'\s+', ' ', str(text).strip())

    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)

    return [s.strip() for s in sentences if len(s.strip()) > 1]

def preprocess_reports(input_csv, output_pkl):
    df = pd.read_csv(input_csv)
    sentences = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        subj_id = row["subject_id"]
        study_id = row["study_id"]
        impression = row.get("impression", "")

        if not isinstance(impression, str) or not impression.strip():
            continue

        for sentence in sentence_split_regex(impression):
            sentences.append({
                "subject_id": subj_id,
                "study_id": study_id,
                "sentence": sentence
            })

    print(f"Extracted {len(sentences)} impression sentences")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(sentences, f)

if __name__ == "__main__":
    preprocess_reports(
        input_csv="../data/radiology_reports.csv",
        output_pkl="../data/preprocessed_sentences.pkl"
    )
