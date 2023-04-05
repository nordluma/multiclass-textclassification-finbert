import csv
import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim.adamw import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("data/preprocessed_data.csv")

# Rename class to label and classes to indexes
possible_labels = df["label"].unique()

label_dict = {}
for idx, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = idx

df["label"] = df["label"].replace(label_dict)

BERT_MODEL = "TurkuNLP/bert-base-finnish-cased-v1"

# Hyper parameters
NUM_EPOCHS = 2
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPSILON = 1e-8
SEED_VAL = 17

# Split data into train/test
X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df["label"].values,
    test_size=0.20,
    random_state=42,
    stratify=df["label"].values,
)

df["data_type"] = ["not-set"] * df.shape[0]

df.loc[X_train, "data_type"] = "train"
df.loc[X_val, "data_type"] = "val"

# Tokenize data
print("Loading tokenizer")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

print("Tokenizing training data")
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == "train"].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

print("Tokenizing validation data")
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == "val"].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

input_ids_train = encoded_data_train["input_ids"]
attention_mask_train = encoded_data_train["attention_mask"]
labels_train = torch.tensor(df[df.data_type == "train"].label.values)


input_ids_val = encoded_data_val["input_ids"]
attention_mask_val = encoded_data_val["attention_mask"]
labels_val = torch.tensor(df[df.data_type == "val"].label.values)

dataset_train = TensorDataset(input_ids_train, attention_mask_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_mask_val, labels_val)

# Load Model
print("Loading BERT model")
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL,
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False,
)

print("")

model = model.to(device)

dataloader_train = DataLoader(
    dataset_train, sampler=RandomSampler(dataset_train), batch_size=BATCH_SIZE
)

dataloader_val = DataLoader(
    dataset_val, sampler=SequentialSampler(dataset_val), batch_size=BATCH_SIZE
)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

TOTAL_STEPS = len(dataloader_train) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=TOTAL_STEPS
)


# Performance metrics
def f1_score_fn(preds, labels):
    """Calculate F1 score"""
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


def flat_acc(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / labels_flat


def acc_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]

        print(f"Class: {label_dict_inverse[label]}")
        print(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}")


# Define train_model function

random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)


def evaluate(dataloader_val):
    model.eval()
    val_total_loss = 0
    # val_total_acc = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        val_total_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()

        predictions.append(logits)
        true_vals.append(label_ids)
        # val_total_acc += flat_acc(predictions, label_ids)

    # val_avg_acc = val_total_acc / len(dataloader_val)
    val_avg_loss = val_total_loss / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return val_avg_loss, predictions, true_vals


print("Starting training loop")
training_stats = []

for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    model.train()
    train_total_loss = 0

    progress_bar = tqdm(
        dataloader_train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
    )

    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = model(**inputs)

        loss = outputs[0]
        train_total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    torch.save(model.state_dict(), f"model/finetuned_finBERT_epoch_{epoch}.model")

    train_avg_loss = train_total_loss / len(dataloader_train)
    val_avg_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_fn(predictions, true_vals)

    tqdm.write(f"\nEpoch: {epoch}")
    tqdm.write(f"Training loss: {train_avg_loss}")
    tqdm.write(f"Validation loss: {val_avg_loss}")
    tqdm.write(f"F1 Score (weighted): {val_f1}")

    # Record stats
    training_stats.append(
        {
            "epoch": epoch,
            "Training loss": train_avg_loss,
            "Validation loss": val_avg_loss,
            "Validation F1": val_f1,
        }
    )

print("")
print("Training completed")

field_names = [
    "epoch",
    "Training loss",
    "Validation loss",
    "Validation F1",
]

print("Writing training stats to file")
with open("data/training_data.csv", "w+") as f:
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(training_stats)

# =========Evaluate the model=========
print("Starting model evaluation")
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL,
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

print("")

model.load_state_dict(
    torch.load(
        "model/finetuned_finBERT_epoch_2.model", map_location=torch.device("cpu")
    )
)

# Test on validation data
print("Testing model with validation data")
_, predictions, true_vals = evaluate(dataloader_val)
acc_per_class(predictions, true_vals)

# Test with test data
print("Testing model with test data")
