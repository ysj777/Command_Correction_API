import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datasets
from datasets import Dataset, DatasetDict
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ast
from transformers import AutoTokenizer,BertTokenizer, BertForTokenClassification, get_scheduler, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.optim import AdamW
# import evaluate
from huggingface_hub import notebook_login
from torch.utils.data import DataLoader
import logging
from torch import cuda
import warnings


# seqeval = evaluate.load("seqeval")

id2label = {
    0: 'O',
    1: 'B-Modify',
    2: 'B-Filling',
}

label2id = {
    'O': 0,
    'B-Modify': 1,
    'B-Filling': 2,
}

label_list = ['O', 'B-Modify', 'B-Filling']

# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3, id2label=id2label, label2id=label2id)

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.dataset[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.dataset[idx]['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.dataset[idx]['labels'], dtype=torch.long)
        }

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def transform_labels(labels):
    return [label2id.get(label, 0) for label in labels]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def create_dataset(data, is_test=False):
    dataset = []
    for _, row in data.iterrows():
        text = row['text']
        label = row['label']
        if not is_test and len(label) != len(tokenizer.tokenize(text)):
            print(f"Label and text length mismatch in row: {row}")
            continue  # 跳過長度不匹配的數據

        tokenized_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256)
        if not is_test:
            label = transform_labels(label)
            word_ids = tokenized_inputs.word_ids()

            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            tokenized_inputs['labels'] = label_ids

        dataset.append({
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': tokenized_inputs.get('labels', None) if not is_test else None
        })

    return dataset


def evaluate_test_data(test_data):
    device = 'cuda' if cuda.is_available() else 'cpu'
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    test_dataset = create_dataset(test_data,is_test=True)
    test_dataloader = CustomDataset(test_dataset)
    model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3, id2label=id2label, label2id=label2id)
    model.load_state_dict(torch.load('model_weight/modify_best_model_state.pt',map_location= device))
    model.to(device)
    model.eval()
    # 生成预测和标签
    true_predictions = []
    true_labels = []

    for batch in test_dataloader:
        predictions = model(batch['input_ids'])
        labels = batch['labels']
        true_predictions.append(predictions)
        true_labels.append(labels)

    # 调整序列长度
    true_predictions = [p[:len(l)] for p, l in zip(true_predictions, true_labels)]
    true_labels = [l[:len(p)] for p, l in zip(true_predictions, true_labels)]

    # 检查长度是否一致
    assert all(len(p) == len(l) for p, l in zip(true_predictions, true_labels)), "预测序列和真实标签长度不一致！"

    # 计算评估指标
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)




def main():
    set_seed(42)
    df = pd.read_csv("modify_araea.csv", encoding="utf-8")
    df['label'] = df['label'].apply(ast.literal_eval)
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    print(train_data.shape, val_data.shape, test_data.shape)
    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    train_dataset = CustomDataset(train_dataset)
    print(train_dataset[0])
    val_dataset = CustomDataset(val_dataset)
    print(val_dataset[0])


    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    training_args = TrainingArguments(
        output_dir="my_awesome_wnut_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    evaluate_test_data(model, tokenizer, test_data, label_list)

class Modify_area_extractor:
    def __init__(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3, id2label=id2label, label2id=label2id)
        #self.model.load_state_dict(torch.load('model_weight/modify_best_model_state.pt', map_location=self.device))
        self.model.load_state_dict(torch.load('model_weight/best_model_state_mix.pt', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    def predict(self, text):
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
            return_offsets_mapping=True,  # 加這行
            return_attention_mask=True
        )

        input_ids = tokenized_input['input_ids'].to(self.device)
        attention_mask = tokenized_input['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = np.argmax(logits.cpu().numpy(), axis=2)

        predicted_labels = []
        word_ids = tokenized_input.word_ids()
        offsets = tokenized_input['offset_mapping'][0].tolist()  # offset for each token
        previous_word_idx = None
        selected_offsets = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                predicted_labels.append(label_list[predictions[0][idx]])
                selected_offsets.append(offsets[idx])
            previous_word_idx = word_idx

        return {
            "labels": predicted_labels,
            "offsets": selected_offsets,
            "text": text
        }


def predict(text):
    device = 'cuda' if cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Tokenize the input text
    tokenized_input = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    # Load the model
    model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3, id2label=id2label, label2id=label2id)
    model.load_state_dict(torch.load('model_weight/modify_best_model_state.pt', map_location=device))
    model.to(device)
    model.eval()

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = np.argmax(logits.cpu().numpy(), axis=2)

    # Map predictions to labels
    predicted_labels = []
    word_ids = tokenized_input.word_ids()
    previous_word_idx = None

    # Aggregate subword predictions back to original words
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:  # Skip padding
            continue
        if word_idx != previous_word_idx:  # New word
            predicted_labels.append(label_list[predictions[0][idx]])
        previous_word_idx = word_idx

    return predicted_labels

if __name__ == '__main__':
    # Example usage
    text_input = "不過話說回來，我覺得有的服務事不應該商業化的。 [SEP] 刪除事情的事"
    predicted_labels = predict(text_input)
    print(predicted_labels)


# if __name__ == '__main__':
#     text2 = "在參後面插入與的與聽起來是一份很好的公司。又意思又很多錢。 [SEP] 請把又意思的又改成有沒有的有,"
#     answer2 = evaluate_test_data(text2)
#     print(answer2)