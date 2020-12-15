#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : spacy_demo.py
# @ Description: https://realpython.com/natural-language-processing-spacy-python/
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/15 上午9:19
# @ Software   : PyCharm
#-------------------------------------------------------

import spacy
from collections import Counter

def main():
    text = 'FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, ' \
           'Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland.Headquartered in Zürich, its ' \
           'membership now comprises 211 national associations. Member countries must each also be members of one of ' \
           'the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America ' \
           'and the Caribbean, Oceania, and South America.'

    nlp = spacy.load('en')
    # doc = nlp(u'This is a sentence.')
    doc =  nlp(text)
    # tokenizer 分词
    # docs = nlp.tokenizer(text)
    print('*' * 20, 'detect sentence', '*' * 20)
    for sentence in doc.sents:
        print(sentence)

    print('*' * 20, 'tokenizer', '*' * 20)
    for token in doc:
        print(token, token.idx)

    # lemmatize 词干化
    print('*' * 20, 'lemmatize', '*' * 20)
    for token in doc:
        print(token, token.lemma_)


    # word frequency
    print('*' * 20, 'word frequency', '*' * 20)

    # Remove stop words and punctuation symbols
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    word_freq = Counter(words)
    # 5 commonly occurring words with their frequencies
    common_words = word_freq.most_common(5)
    print(common_words)

    # Unique words
    unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
    print(unique_words)


    # POS(Part Of Speech) tagging 词性标注
    """
    Noun
    Pronoun
    Adjective
    Verb
    Adverb
    Preposition
    Conjunction
    Interjection
    """

    print('*' * 20, 'POS tagging', '*' * 20)
    for token in doc:
        print(token, token.pos_, token.pos)

    # Named Entity Recognition (NER)
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))
    print('*' * 20, 'NER', '*' * 20)
    for entity in doc.ents:
        print(entity, entity.label_, entity.label)

    print('*' * 20, 'Shallow Parsing', '*' * 20)
    # Noun Phrase Detection
    for chunk in doc.noun_chunks:
        print(chunk)
    # Verb Phase Detection

if __name__ == "__main__":
    main()