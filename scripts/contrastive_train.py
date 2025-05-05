import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
import os

def prepare_contrastive_data(file_path, num_negatives=1):
    """
    Prepares sentence pairs for contrastive learning.
    
    Args:
        file_path (str): Path to labeled sentences pickle file
        num_negatives (int): Number of negative samples per positive
    
    Returns:
        List of InputExample for contrastive training
    """
    df = pd.read_pickle(file_path)
    all_sentences = []
    positives = []
    negatives = []

    for report in df["labeled_sentences"]:
        for entry in report:
            all_sentences.append((entry['text'], entry['label']))
            if entry['label'] == "normal":
                positives.append(entry['text'])
            else:
                negatives.append(entry['text'])

    examples = []

    for text, label in all_sentences:
        if label == "normal":
            positive_example = random.choice(positives)
        else:
            positive_example = random.choice(negatives)
        
        # Positive pair (same label)
        examples.append(InputExample(texts=[text, positive_example], label=1.0))
        
        # Negative pair (different label)
        for _ in range(num_negatives):
            if label == "normal":
                negative_example = random.choice(negatives)
            else:
                negative_example = random.choice(positives)
            examples.append(InputExample(texts=[text, negative_example], label=0.0))

    return examples

def train_contrastive_model(examples, model_name, output_dir, batch_size=16, epochs=1):
    """
    Fine-tunes a SentenceTransformer model with contrastive learning.
    
    Args:
        examples (List[InputExample])
        model_name (str)
        output_dir (str)
        batch_size (int)
        epochs (int)
    """
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_dir
    )

if __name__ == "__main__":
    examples = prepare_contrastive_data("data/labeled_sentences.pkl")
    train_contrastive_model(
        examples,
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", # or your own fine-tuned BioBERT
        output_dir="models/student_contrastive_model",
        batch_size=16,
        epochs=2
    )
