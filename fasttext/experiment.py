import fasttext
import logging
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict


def f1_np(y_true, y_pred):
    """F1 metric.

    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    # macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    if np.sum(predicted_positives) == 0:
        precision = 0.0
    else:
        a = np.sum(predicted_positives)
        b = np.sum(true_positives)
        c = np.sum(possible_positives)
        precision = np.sum(true_positives) / np.sum(predicted_positives)

    recall = np.sum(true_positives) / np.sum(possible_positives)
    # micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall


if __name__ == '__main__':
    train_file = '../data/baidu_95_train_seg.csv'
    valid_file = '../data/baidu_95_valid_seg.csv'

    model = fasttext.train_supervised(input=train_file, lr=1.2,  epoch=50, wordNgrams=3, bucket=2000000, dim=128)
    # model = fasttext.load_model('fasttext_base.bin')
    model.save_model('fasttext_cut_words.bin')
    result = model.test(valid_file)
    label_true, label_pred = [], []
    # 验证模型
    label_result = defaultdict(lambda : [])
    with open(valid_file, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            labels = line[:-1].split()[0]
            string = ''.join(line.split()[1:])
            i += 1
            predicts = model.predict(' '.join(string))
            label_true.append({labels})
            label_pred.append(set(predicts[0]))
            pre = set()
            pre.add(predicts[0][0])
            label_result[labels].append(set(predicts[0]))
    # model.save_model('test_result.model')
    # 评估模型
    # for i in pre:
    #     print(i)
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(label_true)
    y_pred = mlb.transform(label_pred)
    result = f1_np(y_true, y_pred)
    print('precision:', result[0], ' ', 'recall:', result[1])
    # for label in label_result.keys():
    #     batch_y_true = mlb.fit_transform([{label}]*len(label_result[label]))
    #     batch_y_pred = mlb.transform(label_result[label])
    #     batch_result = f1_np(batch_y_true, batch_y_pred)
    #     print(label+':\t'+str(batch_result[1]))
