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
import random
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.vocab import GloVe


device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 2020
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 128
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

def main():
    # -----------------get train, val and test data--------------------
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='../Dataset/IMDB')

    print(train_data.fileds)
    print(train_data.examples[0])

    train_data, val_data = train_data.split(random_state = random.seed(RANDOM_SEED))

    print('Number of train data {}'.format(len(train_data)))
    print('Number of val data {}'.format(len(val_data)))
    print('Number of val data {}'.format(len(test_data)))


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
    val_iter, test_iter = data.BucketIterator((val_data, test_data), batch_size=BATCH_SIZE, device=device,
                                              sort_within_batch=True)

    for batch_data in train_iter:
        print(batch_data.text)
        print(batch_data.label)
        break


    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layer, pad_index,
                     bidirectional=False, dropout=0.5):
            super(BiLSTM, self).__init__()

            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=embedding_dim,
                                          padding_idx=pad_index)
            self.lstm = nn.LSTM(input_size=embedding_dim,
                                hidden_size=hidden_size,
                                bidirectional=bidirectional,
                                dropout=dropout)

            if bidirectional:
                self.fc = nn.Linear(hidden_size * 2,  output_size)
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
            # pack sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length, batch_first=False)

            # lstm
            # h_n => (num_direction * num_layers, batch_size, hidden_size)
            # c_n => (num_direction * num_layers, batch_size, hidden_size)
            packed_output, (h_n, c_n)= self.lstm(packed_embedded)

            # unpacked sequence
            output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)

            # hidden => (batch_size, hidden_size*num_direction)
            hidden = torch.cat((h_n[-2., :, :], h_n[-1., :, :]), dim=1)
            hidden = self.dropout(hidden)

            return hidden


if __name__ == "__main__":
    main()