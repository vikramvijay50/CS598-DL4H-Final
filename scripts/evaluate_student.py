import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

MODEL_DIR = "../models/student_model"
DATA_PATH = "../data/sentences_annotated_teacher.pkl"
BATCH_SIZE = 32
MAX_LENGTH = 128

label2id = {"normal": 0, "abnormal": 1, "uncertain": 2}
id2label = {v: k for k, v in label2id.items()}

class RadiologyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["sentence"]
        label = label2id[item["label"]]

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted = torch.argmax(outputs.logits, dim=1)

            preds.extend(predicted.cpu().tolist())
            labels.extend(true_labels.cpu().tolist())

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")

    print("\nEvaluation Results")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


def main():
    print("Loading annotated data...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    print("Using 20% of data for evaluation...")
    val_split = int(0.8 * len(data))
    val_data = data[val_split:]

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = RadiologyDataset(val_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    print("Starting evaluation...")
    evaluate(model, dataloader, device)


if __name__ == "__main__":
    main()
