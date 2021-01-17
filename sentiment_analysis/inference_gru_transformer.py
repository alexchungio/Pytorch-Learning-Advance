#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference_gru_transformer.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/15 上午9:39
# @ Software   : PyCharm
#-------------------------------------------------------


import torch
import spacy

from sentiment_analysis.gru_with_transformer import BertGRUSentiment
from transformers import BertTokenizer, BertModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = './output/transformer_model_best.pth'

nlp = spacy.load('en')

state = torch.load(MODEL_PATH)
HIDDEN_SIZE = state['hidden_size']
OUTPUT_SIZE = state['output_size']
NUM_LAYERS = state['num_layer']
BIDIRECTIONAL = state['bidirectional']
DROPOUT = state['dropout']


def predict(model, sentence, tokenizer):

    model.eval()
    # step 1 tokenize
    tokens = tokenizer.tokenize(sentence)
    # step 2 get max_length
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    # cut sentence for the following insert head and tail to the text
    tokens = tokens[:max_input_length - 2]
    # token to index
    token_index = tokenizer.convert_tokens_to_ids(tokens)
    # add head and tail
    token_index = [tokenizer.cls_token_id] + token_index + [tokenizer.sep_token_id]
    # (seq_length, )
    text_tensor = torch.LongTensor(token_index).to(device)
    # (seq_length) => (batch_size, seq_length) => (seq_length, batch_size)
    # text_tensor = text_tensor.unsqueeze(dim=0).transpose(1, 0)
    text_tensor = text_tensor.unsqueeze(0)
    pred = torch.sigmoid(model(text_tensor))

    return pred


def main():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertGRUSentiment(bert=bert_model,
                             hidden_size=HIDDEN_SIZE,
                             num_layers=NUM_LAYERS,
                             output_size=OUTPUT_SIZE,
                             bidirectional=BIDIRECTIONAL,
                             dropout=DROPOUT)

    # load trained model
    model.load_state_dict(state['state_dict'])
    model = model.to(device)

    sentence_0 = 'This film is terrible'
    pred_0 = predict(model, sentence_0, tokenizer)
    print(pred_0.item()) # 0.031715769320726395

    sentence_1 = 'This film is great'
    pred_1 = predict(model, sentence_1, tokenizer)
    print(pred_1.item()) # 0.9611917734146118


if __name__ == "__main__":
    main()