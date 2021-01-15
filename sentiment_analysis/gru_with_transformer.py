#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : gru_with_transformer.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/15 上午9:38
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import numpy as np
import random
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext import data, datasets

from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import shutil


device = 'cuda' if torch.cuda.is_available() else 'cpu'


SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.cuda.empty_cache()  # release gpu capacity

class BertGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_size, output_size, num_layers, bidirectional=False, dropout=0.5):
        super(BertGRUSentiment, self).__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if num_layers < 2 else dropout)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, text):

        # sentence => (batch_size, length_sentence)
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded => （batch_size, length, embedding_dim）
        _, hidden = self.gru(embedded)

        # hidden => (num_layers * num_directions, batch size, embedding_dim)
        # only use hidden state of last layer
        if self.gru.bidirectional:
            last_hidden = torch.cat((hidden[-2, : ,:], hidden[-1, :, :]), dim=1)

        else:
            last_hidden = hidden[-1, :, :]

        # last_hidden => (batch_size, num_direction * embedding_dim)

        # dropout
        last_hidden = self.dropout(last_hidden)

        # output => (batch_size, output_size)
        outputs = self.fc(last_hidden)

        return outputs


def get_field(tokenizer):

    # step 1 get special tokens indices
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

    # step 2 get max_length
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    print('max input size {}'.format(max_input_length))

    # step 3 define tokenize
    def tokenize_with_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]

        return tokens

    # step 4 define field
    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_with_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    LABEL = data.LabelField(dtype=torch.float)

    return TEXT, LABEL


def count_trainable_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(pred, target, threshold=0.5):

    preds = torch.sigmoid(pred) > threshold

    correct = (preds==target).float()

    return correct.mean()


def train(model, data_loader, optimizer, criterion):
    model.train()

    epoch_loss = []
    epoch_acc = []

    pbar = tqdm(data_loader)
    for data in pbar:
        text = data.text.to(device)  # (batch_size, max_input_length)
        label = data.label.to(device) # (batch_size, )

        pred = model(text).squeeze(dim=1)  # (batch_size, 1) => (batch_size, )
        loss = criterion(pred, label)
        # grad clearing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = binary_accuracy(pred, label).item()
        batch_size = text.shape[0]

        epoch_loss.append(batch_loss)
        epoch_acc.append(batch_acc)

        pbar.set_description('train => acc {} loss {}'.format(batch_acc, batch_loss))

    return sum(epoch_acc) / len(data_loader), sum(epoch_loss) / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()

    epoch_loss = []
    epoch_acc = []
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for data in pbar:
            text = data.text.to(device)
            label = data.label.to(device)

            pred = model(text).squeeze(dim=1)
            loss = criterion(pred, label)

            batch_loss = loss.item()
            batch_acc = binary_accuracy(pred, label).item()
            batch_size = text.shape[0]

            epoch_loss.append(batch_loss)
            epoch_acc.append(batch_acc)

            pbar.set_description('eval => acc {} loss {}'.format(batch_acc, batch_loss))

    return sum(epoch_acc) / len(data_loader), sum(epoch_loss) / len(data_loader)


def main():

    BATCH_SIZE = 32
    MODEL_PATH = './output/transformer_model.pth'
    BEST_MODEL_PATH = './output/transformer_model_best.pth'
    # ---------------------------define field-------------------------
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    TEXT, LABEL = get_field(tokenizer)

    # -----------------get train, val and test data--------------------
    # load the data and create the validation splits
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='../Dataset/IMDB')

    LABEL.build_vocab(train_data)
    print(LABEL.vocab.stoi)

    train_data, eval_data = train_data.split(random_state = random.seed(SEED))

    print('Number of train data {}'.format(len(train_data)))
    print('Number of evaluate data {}'.format(len(eval_data)))
    print('Number of test data {}'.format(len(test_data)))


    # generate dataloader
    train_iterator, eval_iterator = data.BucketIterator.splits((train_data, eval_data),
                                                                batch_size=BATCH_SIZE,
                                                                device=device)

    test_iterator = data.BucketIterator(test_data, batch_size=BATCH_SIZE, device=device)

    for batch_data in train_iterator:
        print('text size {}'.format(batch_data.text.size()))
        print('label size {}'.format(batch_data.label.size()))
        break

    # --------------------------------- build model -------------------------------
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 1
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertGRUSentiment(bert=bert_model,
                             hidden_size=HIDDEN_SIZE,
                             num_layers=NUM_LAYERS,
                             output_size=OUTPUT_SIZE,
                             bidirectional=BIDIRECTIONAL,
                             dropout=DROPOUT)

    # frozen bert
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
    # get model trainable parameter
    print('The model has {:,} trainable parameters'.format(count_trainable_parameters(model)))

    # check trainable parameter
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # ---------------------------------- config -------------------------------------------

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    # ----------------------------------- train -------------------------------------------
    NUM_EPOCH = 10
    model = model.to(device)

    # train and evalate
    best_eval_loss = float('inf')
    for epoch in range(NUM_EPOCH):
        print('{}/{}'.format(epoch, NUM_EPOCH))
        train_acc, train_loss = train(model, train_iterator, optimizer=optimizer, criterion=criterion)
        eval_acc, eval_loss = evaluate(model, eval_iterator, criterion=criterion)
        scheduler.step()
        print('Train => acc {:.3f}, loss {:4f}'.format(train_acc, train_loss))
        print('Eval => acc {:.3f}, loss {:4f}'.format(eval_acc, eval_loss))

        # save model
        state = {
            'hidden_size': HIDDEN_SIZE,
            'output_size': OUTPUT_SIZE,
            'num_layer': NUM_LAYERS,
            'bidirectional': BIDIRECTIONAL,
            'dropout': DROPOUT,
            'state_dict': model.state_dict(),
        }
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(state, MODEL_PATH)
        if eval_loss < best_eval_loss:
            shutil.copy(MODEL_PATH, BEST_MODEL_PATH)
            best_eval_loss = eval_loss

    # test
    test_acc, test_loss = evaluate(model, test_iterator, criterion)
    print('Test => acc {:.3f}, loss {:4f}'.format(test_acc, test_loss))


if __name__ == "__main__":
    main()