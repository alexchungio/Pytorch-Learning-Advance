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

import torch
import torch.nn as nn
import torchtext
import torchtext.data as data


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


def main():

    # ---------------------------- config--------------------------------------
    num_epochs = 200
    seq_length = 32
    time_step = 40
    clip_theta = 0.001

    dataset_mode = 'random'  # random | consecutive



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

    print(tokenizer.vocab_size)

    corpus_indices = tokenizer.text_to_array(corpus_chars)

    print(corpus_indices[:40])
    print(''.join([index_to_char[index] for index in corpus_indices[:40]]))



if __name__ == "__main__":

    main()




