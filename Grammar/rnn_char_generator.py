#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rnn_char_generator.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/27 上午11:02
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


def data_loader_random(corpus_indices, seq_length, time_steps, device=None):
    num_samples = (len(corpus_indices) - 1) // time_steps
    epoch_size = num_samples // seq_length

    # shuffle sample
    samples_indices = list(range(num_samples))
    random.shuffle(samples_indices)

    # generator data
    for i in range(epoch_size):
        i = i * seq_length

        batch_incices = samples_indices[i: i + seq_length]
        x = [corpus_indices[indices: indices + time_steps] for indices in batch_incices]
        y = [corpus_indices[indices + 1: indices + time_steps + 1] for indices in batch_incices]

        x = torch.tensor(x, dtype=torch.float32).view(seq_length, time_steps)
        y = torch.tensor(y, dtype=torch.float32).view(seq_length, time_steps)

        yield x, y


def data_loader_consecutive(corpus_indices, seq_length, time_steps):

    data_len = len(corpus_indices)

    seq_size = data_len // seq_length

    # resize to => (seq_length, seq_size)
    corpus_indices = np.array(corpus_indices[: seq_size * seq_length], dtype=np.float).reshape((seq_length, -1))

    epoch_size = (seq_size - 1) // time_steps

    # generator data
    np.random.shuffle(corpus_indices)

    # convert to torch tensor
    torch_indices = torch.tensor(corpus_indices, dtype=torch.float32).view(seq_length, seq_size)
    for i in range(epoch_size):
        i = i * time_steps
        x = torch_indices[:, i: i + time_steps]
        y = torch_indices[:, i + 1: i + time_steps + 1]

        yield x, y

def onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)

    return [F.one_hot(X[:, i].to(torch.int64), n_class).to(dtype=torch.float32, device=device) for i in range(X.shape[1])]

 # class RNNCell(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super(RNNCell, self).__init__()
    #
    #         self.hidden_size = hidden_size
    #
    #         self.w_ih = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(input_size, hidden_size)),
    #                                               dtype=torch.float32),
    #                                  requires_grad=True)
    #         self.w_hh = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(hidden_size, hidden_size)),
    #                                               dtype=torch.float32),
    #                                  requires_grad=True)
    #         self.b_h = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
    #
    #         self.w_hq = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(hidden_size, output_size)),
    #                                               dtype=torch.float32),
    #                                  requires_grad=True)
    #         self.b_q = nn.Parameter(torch.zeros(output_size), requires_grad=True)
    #
    #     def forward(self, inputs, state):
    #         """
    #
    #         :param inputs:
    #         :return:
    #         """
    #         # initial state
    #         # h_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.double)
    #         outputs = []
    #
    #         h , = state  # hidden state
    #         for input in inputs:
    #             h = torch.tanh(
    #                 torch.matmul(input, self.w_ih) + torch.matmul(h, self.w_hh) + self.b_h)
    #
    #             out = torch.matmul(h, self.w_hq) + self.b_q
    #
    #             outputs.append(out)
    #
    #         return outputs, (h, )


class RNNModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim, bidirectional=False):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=self.hidden_size, bidirectional=bidirectional, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)


        x = self.embedding(inputs.long())
        out, self.state = self.rnn(x, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(out.contiguous().view(out.shape[0]*out.shape[1], out.shape[-1]))
        return output, self.state


def predict(model, prefix, num_chars, device, index_to_char, char_to_index):

    model.to(device)
    state = None
    output = [char_to_index[prefix[0]]] # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_index[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([index_to_char[i] for i in output])


def train_and_predict(model, device, corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len, prefixes, ):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict(model,
                    prefix, pred_len, device, idx_to_char, char_to_idx))


def main():

    # ---------------------------- config--------------------------------------

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


    model = RNNModel(hidden_size=128, vocab_size=vocab_size, embedding_dim=256)

    for name, param in model.named_parameters():
        print(name, param.size())


    inputs = torch.arange(10).view(2, 5)


    states = None
    outputs, states = model(inputs, states)


    p = predict(model, '分开', 10, device, index_to_char, char_to_index)

    print(p)

    time_step = 40
    # dataset_mode = 'random'  # random | consecutive
    num_epochs, seq_length, lr, clipping_theta = 10000, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict(model, device, corpus_indices, index_to_char, char_to_index,
                      num_epochs, time_step, lr, clipping_theta, seq_length, pred_period, pred_len, prefixes)




if __name__ == "__main__":

    main()

    # x = torch.tensor([1, 6])
    # f = F.one_hot(x, 10)
    # print(f)




