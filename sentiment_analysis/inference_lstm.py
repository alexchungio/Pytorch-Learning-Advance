#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference_lstm.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/15 下午2:35
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import spacy

from sentiment_analysis.bidirected_lstm import BiLSTMSentiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = './output/model.pth'

nlp = spacy.load('en')


state = torch.load(MODEL_PATH)
VOCAB_SIZE = state['vocab_size']
EMBEDDING_DIM = state['embedding_dim']
HIDDEN_SIZE = state['hidden_size']
OUTPUT_SIZE = state['output_size']
NUM_LAYER = state['num_layer']
BIDIRECTIONAL = state['bidirectional']
DROPOUT = state['dropout']
PAD_INDEX = state['pad_index']
TEXT_VOCAB = state['text_vocab']


def predict(model, sentence, text_vocab):

    model.eval()
    tokenized = [token.text for token in nlp.tokenizer(sentence)]
    token_index = [text_vocab[token] for token in tokenized]
    length = [len(token_index)]

    # (seq_length, )
    text_tensor = torch.LongTensor(token_index).to(device)
    # (seq_length) => (batch_size, seq_length) => (seq_length, batch_size)
    text_tensor = text_tensor.unsqueeze(dim=0).transpose(1, 0)
    length_tensor = torch.LongTensor(length)

    pred = torch.sigmoid(model(text_tensor, length_tensor))

    return pred


def main():

    model = BiLSTMSentiment(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                            output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                            dropout=DROPOUT, pad_index=PAD_INDEX)

    # load trained model
    model.load_state_dict(state['state_dict'])
    model = model.to(device)

    sentence_0 = 'This film is terrible'
    pred_0 = predict(model, sentence_0, text_vocab=TEXT_VOCAB)
    print(pred_0.item())  # 0.08206960558891296

    sentence_1 = 'This film is great'
    pred_1 = predict(model, sentence_1, text_vocab=TEXT_VOCAB)
    print(pred_1.item())  # 0.08206960558891296


if __name__ == "__main__":
    main()