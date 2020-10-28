#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : lstm_char_generator.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/28 下午3:28
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import pickle
import zipfile
import copy
import random
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import Grammar.utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextTokenizer(object):
    def __init__(self, text=None, max_vocab=None, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            corpus_chars = set(text)
            # max_vocab_process
            vocab_count = {}
            for word in corpus_chars:
                if vocab_count.get(word) is None:
                    vocab_count[word] = 1
                else:
                    vocab_count[word] += 1

            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            # sort count with number
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if max_vocab is not None and len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.char_to_index = {c: i for i, c in enumerate(self.vocab)}
        self.index_to_char = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_index(self, word):
        if word in self.char_to_index:
            return self.char_to_index[word]
        else:
            return len(self.vocab)

    def index_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.char_to_index[index]
        else:
            raise Exception('Unknown index!')

    def text_to_array(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_index(word))
        return np.array(arr)

    def array_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.index_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)


def data_loader_random(corpus_indices, seq_len, time_steps, device=None):
    num_samples = (len(corpus_indices) - 1) // time_steps
    epoch_size = num_samples // seq_len

    # shuffle sample
    samples_indices = list(range(num_samples))
    random.shuffle(samples_indices)

    # generator data
    for i in range(epoch_size):
        i = i * seq_len

        batch_incices = samples_indices[i: i + seq_len]
        x = [corpus_indices[indices: indices + time_steps] for indices in batch_incices]
        y = [corpus_indices[indices + 1: indices + time_steps + 1] for indices in batch_incices]

        x = torch.tensor(x, dtype=torch.float32).view(seq_len, time_steps)
        y = torch.tensor(y, dtype=torch.float32).view(seq_len, time_steps)

        yield x, y


def data_loader_consecutive(corpus_indices, seq_len, time_steps):

    data_len = len(corpus_indices)

    seq_size = data_len // seq_len

    # resize to => (seq_len, seq_size)
    corpus_indices = np.array(corpus_indices[: seq_size * seq_len], dtype=np.float).reshape((seq_len, -1))

    epoch_size = (seq_size - 1) // time_steps

    # generator data
    np.random.shuffle(corpus_indices)

    # convert to torch tensor
    torch_indices = torch.tensor(corpus_indices, dtype=torch.float32).view(seq_len, seq_size)
    for i in range(epoch_size):
        i = i * time_steps
        x = torch_indices[:, i: i + time_steps]
        y = torch_indices[:, i + 1: i + time_steps + 1]

        yield x, y

def onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)

    return [F.one_hot(X[:, i].to(torch.int64), n_class).to(dtype=torch.float32, device=device) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


class LSTMModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim, bidirectional=False):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, bidirectional=bidirectional,
                           batch_first=True)
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (seq_len, time_steps)

        # (seq_len, time_steps, embedding_dim)
        x = self.embedding(inputs.long())
        # (seq_len, time_steps, vocab_size)
        out, self.state = self.lstm(x, state)
        # (seq_len*time_step, vocab_size)
        output = self.dense(out.contiguous().view(-1, out.shape[-1]))
        return output, self.state


def predict(model, prefix, num_chars, device, index_to_char, char_to_index):

    state=None
    output = [char_to_index[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # read one char as input
        x = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (y, state) = model(x, state)
        if t < len(prefix) - 1:
            output.append(char_to_index[prefix[t + 1]])
        else:
            output.append(int(y.argmax(dim=1).item()))
    return ''.join([index_to_char[i] for i in output])


def train_and_predict(model, corpus_indices, sample_model, num_epochs, seq_len, time_steps, lr, clipping_theta,
                      index_to_char, char_to_index, pred_period, pred_len, prefixes, device):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        state = None  # state
        if sample_model == 'consecutive':
            # state = None  # reset state at every epoch
            data_loader = data_loader_consecutive(corpus_indices, seq_len=seq_len, time_steps=time_steps)
        else:
            data_loader = data_loader_random(corpus_indices, seq_len=seq_len, time_steps=time_steps)

        l_sum, n, start = 0.0, 0, time.time()
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            if sample_model == "random":
                state = None  # reset state at every step
            else:
                if state is not None:
                    # pytorch是动态计算图，每次backward后，本次计算图自动销毁，但是计算图中的节点都还保留。
                    # 方向传播直到叶子节点为止，否者一直传播，直到找到叶子节点
                    # 这里 detach的作用是梯度节流，防止反向传播传播到隐藏状态时，因为上次小批量方向传播计算图的销毁导致继续向下传播而引起报错。
                    if isinstance (state, tuple): # LSTM, state:(h, c)
                        state = (state[0].detach(), state[1].detach())
                    else:
                        state = state.detach()

            (output, state) = model(x, state) # output: 形状为(num_steps * batch_size, vocab_size)

            # (seq_len, time_steps) => (seq_len * time_steps, 1)
            y = y.contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # gradient clip
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        # computer perplexity
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict(model, prefix, pred_len, device, index_to_char, char_to_index))


def main():

    #------------------------------ load dataset-------------------------------
    with zipfile.ZipFile('../Datasets/jaychou_lyrics/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')  # corpus

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\t', ' ')
    corpus_chars = corpus_chars[: 20000]

    tokenizer = TextTokenizer(text=corpus_chars, max_vocab=None)

    vocab = tokenizer.vocab
    char_to_index = tokenizer.char_to_index
    index_to_char = tokenizer.index_to_char
    vocab_size = tokenizer.vocab_size

    print(tokenizer.vocab_size)

    corpus_indices = tokenizer.text_to_array(corpus_chars)

    print(corpus_indices[:40])
    print(''.join([index_to_char[index] for index in corpus_indices[:40]]))


    # ------------------------- config----------------------------
    embedding_dim = 256
    hidden_size = 512
    seq_len = 32
    time_steps = 40

    num_epochs = 1000
    lr = 0.001
    clipping_theta = 0.01

    sample_mode = 'consecutive'  # random | consecutive

    pred_period, pred_len, prefixes = 50, 100, ['好想', '不想']


    model = LSTMModel(hidden_size=hidden_size, vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)

    for name, param in model.named_parameters():
        print(name, param.size())

    train_and_predict(model=model, corpus_indices=corpus_indices, sample_model=sample_mode, num_epochs=num_epochs,
                      seq_len=seq_len, time_steps=time_steps, lr=lr, clipping_theta=clipping_theta,
                      index_to_char=index_to_char, char_to_index=char_to_index, pred_period=pred_period,
                      pred_len=pred_len, prefixes=prefixes, device=device)


if __name__ == "__main__":
    main()
