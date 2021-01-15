#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : bidirected_lstm.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/14 下午2:09
# @ Software   : PyCharm
#-------------------------------------------------------

import os

import random
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torchtext import data, datasets
from torchtext.vocab import GloVe
import spacy
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 2020
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 128
MODEL_PATH = './output/bilstm_model.pth'


torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# define datatype
# method 1
# python -m spacy download en
# method 2
# step 1: manual download from https://github-production-release-asset-2e65be.s3.amazonaws.com/84940268/69ded28e-c3ef-11e7-94dc-d5b03d9597d8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20201214%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201214T064457Z&X-Amz-Expires=300&X-Amz-Signature=631b41e8491a84dfb7c492f336d728f116a04f677c33cf709dd719d5cf4c126f&X-Amz-SignedHeaders=host&actor_id=26615837&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_core_web_sm-2.0.0.tar.gz&response-content-type=application%2Foctet-stream
# step 2: remove to /home/alex/anaconda3/envs/pytorch/lib/python3.6/site-packages/spacy/data
# step 3: $ pip install en_core_web_sm-2.0.0.tar.gz
# step 4: $ spacy link en_core_web_sm en

# TEXT = data.Field(tokenize='spacy', fix_length=1000)
TEXT = data.Field(tokenize='spacy', include_lengths=True)
LABEL = data.LabelField(sequential=False, dtype=torch.float32)
nlp = spacy.load('en')

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layer, pad_index,
                 bidirectional=False, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=pad_index)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            bidirectional=bidirectional,
                            dropout=dropout)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text, text_length):
        """

        :param text: (seq_len, batch_size)
        :param text_length:
        :return:
        """
        # embedded => [seq_len, batch_size, embedding_dim]
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        text_length = text_length.cpu()  # compatible torch=1.7.0
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length, batch_first=False, enforce_sorted=False)

        # lstm
        # h_n => (num_direction * num_layers, batch_size, hidden_size)
        # c_n => (num_direction * num_layers, batch_size, hidden_size)
        packed_output, (h_n, c_n) = self.lstm(packed_embedded)

        # unpacked sequence
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)

        # hidden => (batch_size, hidden_size*num_direction)
        # only use hidden state of last layer
        if self.lstm.bidirectional:
            hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        else:
            hidden = h_n[-1:, :, :]

        hidden = self.dropout(hidden)

        out = self.fc(hidden)
        return out


def binary_accuracy(pred, target, threshold=0.5):

    preds = torch.sigmoid(pred) > threshold

    correct = (preds==target).float()

    return correct.sum()

def train(model, data_loader, optimizer, criterion):
    model.train()

    epoch_loss = []
    epoch_acc = []
    num_samples = 0

    pbar = tqdm(data_loader)
    for data in pbar:
        text = data.text[0].to(device)
        text_length = data.text[1]
        label = data.label.to(device)

        pred = model(text, text_length)
        loss = criterion(pred, label.unsqueeze(dim=1))
        # grad clearing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = binary_accuracy(pred, label.unsqueeze(dim=1)).item()
        batch_size = text.shape[1]

        epoch_loss.append(batch_loss)
        epoch_acc.append(batch_acc)
        num_samples += batch_size # add batch size

        pbar.set_description('train => acc {} loss {}'.format(batch_acc / batch_size, batch_loss / batch_size))

    return sum(epoch_acc) / num_samples, sum(epoch_loss) / num_samples


def test(model, data_loader, criterion):
    model.eval()

    epoch_loss = []
    epoch_acc = []
    num_samples = 0
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for data in pbar:
            text = data.text[0].to(device)
            text_length = data.text[1]
            label = data.label.to(device)
            pred = model(text, text_length)
            loss = criterion(pred, label.unsqueeze(dim=1))

            batch_loss = loss.item()
            batch_acc = binary_accuracy(pred, label.unsqueeze(dim=1)).item()
            batch_size = text.shape[1]

            epoch_loss.append(batch_loss)
            epoch_acc.append(batch_acc)
            num_samples += batch_size  # add batch size

            pbar.set_description('test => acc {} loss {}'.format(batch_acc / batch_size, batch_loss / batch_size))

    return sum(epoch_acc) / num_samples, sum(epoch_loss) / num_samples



def main():
    # -----------------get train, val and test data--------------------
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='../Dataset/IMDB')

    print(train_data.fileds)
    print(train_data.examples[0])

    train_data, eval_data = train_data.split(random_state = random.seed(RANDOM_SEED))

    print('Number of train data {}'.format(len(train_data)))
    print('Number of val data {}'.format(len(eval_data)))
    print('Number of test data {}'.format(len(test_data)))


    # -------------------initial vocabulary with GLOVE model---------------------------
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=GloVe(name='6B', dim=100, cache='../Dataset/GloVe'),
                     min_freq=10)

    LABEL.build_vocab(train_data)
    print('Unique token in Text vocabulary {}'.format(len(TEXT.vocab))) # 250002(<unk>, <pad>)
    print(TEXT.vocab.itos)
    print('Unique token in LABEL vocabulary {}'.format(len(LABEL.vocab)))
    print(TEXT.vocab.itos)

    print('Top 20 frequency of word: \n {}'.format(TEXT.vocab.freqs.most_common(20)))
    print('Embedding shape {}'.format(TEXT.vocab.vectors.size))

    print('Done')


    # generate dataloader
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=device, shuffle=True)
    val_iter, test_iter = data.BucketIterator.splits((val_data, test_data), batch_size=BATCH_SIZE, device=device,
                                              sort_within_batch=True)

    for batch_data in train_iter:
        print(batch_data.text)  # text, text_length
        print(batch_data.label) # label
        break

    # construct model
    VOCAB_SIZE = len(TEXT.vocab)
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 1
    NUM_LAYER = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    EMBEDDING_DIM = 100
    PAD_INDEX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_INDEX = TEXT.vocab.stoi[TEXT.unk_token]


    model = BiLSTMSentiment(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                            output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                            dropout=DROPOUT, pad_index=PAD_INDEX)


    # load pretrained weight of embedding layer
    pretrained_embedding = TEXT.vocab.vectors
    print(pretrained_embedding)
    pretrained_embedding[PAD_INDEX] = 0
    pretrained_embedding[UNK_INDEX] = 0
    print(pretrained_embedding)

    model.embedding.weight.data.copy_(pretrained_embedding)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # criterion
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = model.to(device)
    EPOCH = 10

    for epoch in range(EPOCH):
        print('{}/{}'.format(epoch, EPOCH))
        train_acc, train_loss = train(model, train_iter, optimizer=optimizer, criterion=criterion)
        test_acc, test_loss = test(model, test_iter, criterion=criterion)

        print('Train => acc {:.3f}, loss {:4f}'.format(train_acc, train_loss))
        print('Eval => acc {:.3f}, loss {:4f}'.format(test_acc, test_loss))
        scheduler.step(epoch=epoch)


    state = {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_size': HIDDEN_SIZE,
        'output_size': OUTPUT_SIZE,
        'num_layer': NUM_LAYER,
        'bidirectional': BIDIRECTIONAL,
        'dropout': DROPOUT,
        'state_dict': model.state_dict(),
        'pad_index': PAD_INDEX,
        'unk_index': UNK_INDEX,
        'text_vocab': TEXT.vocab.stoi,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(state, MODEL_PATH)


if __name__ == "__main__":
    main()