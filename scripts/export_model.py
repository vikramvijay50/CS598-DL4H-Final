import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_DIR = "../models/student_model"
EXPORT_DIR = "../models/exports"
MAX_LENGTH = 128
EXPORT_ONNX = True
EXPORT_TORCHSCRIPT = True
DUMMY_INPUT_TEXT = "The lungs are clear. No acute findings."

os.makedirs(EXPORT_DIR, exist_ok=True)

def export_model():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Tokenize dummy input
    tokens = tokenizer(
        DUMMY_INPUT_TEXT,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    if EXPORT_TORCHSCRIPT:
        print("Exporting to TorchScript...")
        traced_model = torch.jit.trace(model, (input_ids, attention_mask), strict=False)
        torch.jit.save(traced_model, os.path.join(EXPORT_DIR, "student_model.pt"))
        print("TorchScript model saved as student_model.pt")

    if EXPORT_ONNX:
        print("Exporting to ONNX...")
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            os.path.join(EXPORT_DIR, "student_model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"}
            },
            opset_version=14
        )
        print("ONNX model saved as student_model.onnx")

if __name__ == "__main__":
    export_model()
