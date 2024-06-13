import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import fsspec
from alluxiofs import AlluxioFileSystem


class BertDataset(Dataset):
    def __init__(self, tokenizer, max_length, file_path):
        super(BertDataset, self).__init__()

        self.file_path = file_path
        self.train_csv = self.load_data()
        self.tokenizer = tokenizer
        self.target = self.train_csv.iloc[:, 1]
        self.max_length = max_length
        # a list of files
        # when get item, look at list of dataframe, then trigger load data, when exhuasted dataframe, trigger load data
        # size of dataframe - calculate file size, memory size

    def load_data(self):
        fsspec.register_implementation("alluxiofs", AlluxioFileSystem, clobber=True)
        alluxio_fs = fsspec.filesystem("alluxiofs", etcd_hosts="localhost", etcd_port=2379, target_protocol="s3")

        with alluxio_fs.open(self.file_path, mode='r') as f:
            data = pd.read_csv(f, delimiter='\t', header=None)
            print(data)
            return data

    def __len__(self):
        return len(self.train_csv)

    # distributed multiprocess training, what's index in each thread. Is it each load full dataset from s3?
    def __getitem__(self, index):
        # print(index)
        text1 = self.train_csv.iloc[index, 0]

        inputs = self.tokenizer.encode_plus(
            text1,
            None,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.train_csv.iloc[index, 1], dtype=torch.long)
        }


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        out = self.out(o2)

        return out


def finetune(epochs, dataloader, model, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):
        print(epoch)

        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for batch, dl in loop:
            ids = dl['ids']
            token_type_ids = dl['token_type_ids']
            mask = dl['mask']
            label = dl['target']
            label = label.unsqueeze(1)

            optimizer.zero_grad()

            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()

            pred = np.where(output >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct / num_samples

            print(
                f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

            loop.set_description(f'Epoch={epoch+1}/{epochs}')
            loop.set_postfix(loss=loss.item(), acc=accuracy)

    return model


tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

dataset = BertDataset(tokenizer, max_length=100, file_path='s3://sibyltest/BERT_test/sentiment_train.tsv')

dataloader = DataLoader(dataset=dataset, batch_size=32)
model = BERT()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for param in model.bert_model.parameters():
    param.requires_grad = False

model = finetune(2, dataloader, model, loss_fn, optimizer)