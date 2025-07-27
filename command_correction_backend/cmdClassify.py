import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import logging
from torch import cuda


class SentimentData(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = text
        self.text = text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.l1 = BertModel.from_pretrained("bert-base-chinese")
        self.pre_classifier = torch.nn.Linear(768, 256)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(256, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class CMDClassifier:
    def __init__(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        logging.basicConfig(level=logging.ERROR)
        warnings.filterwarnings("ignore")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = ModelClass()
        
        # 正确加载模型权重
        #model_state_dict = torch.load('model_weight/new_berte3_1_state_dict.pt',map_location=torch.device('cpu'))
        model_state_dict = torch.load('model_weight/cmd_classify_bert_mix.pt',map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

    def predict(self, text):
        test_data = SentimentData(text, self.tokenizer, max_len=512)
        test_params = {'batch_size': 1,
                       'shuffle': False,
                       'num_workers': 0
                       }
        testing_loader = DataLoader(test_data, **test_params)
        self.model.eval()
        with torch.no_grad():
            for data in testing_loader:
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids).squeeze()
                answer = torch.argmax(outputs.data)
                return answer.item()



def generate_answer(text):
    device = 'cuda' if cuda.is_available() else 'cpu'
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    test_data = SentimentData(text, tokenizer, max_len=512)
    test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }
    testing_loader = DataLoader(test_data, **test_params)
    model = ModelClass()
    model.load_state_dict(torch.load('model_weight/new_berte3_1_state_dict.pt',map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in testing_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            answer = torch.argmax(outputs.data)
            return answer.item()

if __name__ == '__main__':
    text2 = "在參後面插入與的與"
    answer2 = generate_answer(text2)
    print(answer2)