def replace_data(path, output):
    with open(path, 'r', encoding='utf-8') as f, open(output, 'w', encoding='utf-8') as f2:
        for line in f.readlines():
            data = line.split('\t')
            f2.write(data[0]+'\n')


if __name__ == '__main__':
    replace_data('neg_words.txt', 'negWords.txt')