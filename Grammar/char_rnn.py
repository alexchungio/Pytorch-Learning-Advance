#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : char_rnn.py
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

class TextTokenizer(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print(len(vocab))
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
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.char_to_index = {c: i for i, c in enumerate(self.vocab)}
        self.index_to_char = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.char_to_index:
            return self.char_to_index[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.char_to_index[index]
        else:
            raise Exception('Unknown index!')

    def text_to_array(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def array_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)


# vocab count
def vocab_count(text, max_vocab=None):
    vocab = set(corpus_chars)
    vocab_counts = {}

    # initial vocab_count
    for word in corpus_chars:
        if vocab_counts.get(word) is None:
            vocab_counts[word] = 1
        else:
            vocab_counts[word] += 1

    vocab_count_list = []
    for word, count in vocab_counts.items():
        vocab_count_list.append((word, count))

    # sort according to word count from large to small
    vocab_count_list.sort(key=lambda x: x[1], reverse=True)

    if max_vocab is not None and len(vocab_count_list) > max_vocab:
        vocab_count_list = vocab_count_list[:max_vocab]

    vocab = [x[0] for x in vocab_count_list]

    return vocab


if __name__ == "__main__":
    with zipfile.ZipFile('../Datasets/jaychou_lyrics/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')  # corpus

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\t', ' ')
    corpus_chars = corpus_chars[: 20000]

    vocab = vocab_count(corpus_chars)
    print(vocab)