# -*- coding: utf-8 -*-
import os

import fasttext
import jieba
import numpy as np
import tqdm






def get_data():
    # 只需保证通过get_data，得到一个所有内容分词后的txt即可，词与词之间空格间隔
    with open("finance_news_cut.txt", "w", encoding='utf-8') as f:
        pass

def train_model():
    # 训练词向量模型并保存
    model = fasttext.train_unsupervised('finance_news_cut.txt', )
    model.save_model("news_fasttext.model.bin")

def get_word_vector(word):
    # 获取某词词向量
    model = fasttext.load_model('news_fasttext.model.bin')
    word_vector = model.get_word_vector(word)
    return word_vector

def get_sentence_vector(sentence):
    # 获取某句句向量
    cut_words = jieba.lcut(sentence)
    sentence_vector = None
    for word in cut_words:
        word_vector = get_word_vector(word)
        if sentence_vector is not None:
            sentence_vector += word_vector
        else:
            sentence_vector = word_vector
        sentence_vector = sentence_vector / len(cut_words)
        return sentence_vector

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


if __name__ == "__main__":
    a = get_sentence_vector("可以包邮吗")
    b = get_sentence_vector("能不能包邮")
    print(cos_sim(a, b))
