# data_pre.py

import csv
import torch
from setting import BATCH_SIZE, UNK, PAD, DEVICE
from nltk import word_tokenize
from collections import Counter
import numpy as np
from utils import subsequent_mask, seq_padding
from torch.autograd import Variable

"""
数据预处理：
    输入：数据文件，格式为 "英文文本\t旁遮普文本"
"""

class PrepareData:
    def __init__(self, train_file, dev_file):
        # 读取数据 并分词
        self.train_en, self.train_pu = self.load_data(train_file)
        self.dev_en, self.dev_pu = self.load_data(dev_file)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.pu_word_dict, self.pu_total_words, self.pu_index_dict = self.build_dict(self.train_pu)

        self.train_en, self.train_pu = self.wordToID(self.train_en, self.train_pu, self.en_word_dict, self.pu_word_dict)
        self.dev_en, self.dev_pu = self.wordToID(self.dev_en, self.dev_pu, self.en_word_dict, self.pu_word_dict)

        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_pu, BATCH_SIZE)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_pu, BATCH_SIZE)

    def load_data(self, path):
        """
        读取翻译前(英文)和翻译后(旁遮普)的数据文件
        每条数据都进行分词，然后构建成包含起始符(BOS)和终止符(EOS)的单词列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                pu = [['BOS', 'ਮੈਂ', 'ਤੈਨੂੰ', 'ਪਿਆਰ', 'ਕਰਦਾ', 'ਹਾਂ', 'EOS'], ...]
        """
        en = []
        pu = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')

                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                pu.append(["BOS"] + word_tokenize(line[1]) + ["EOS"])

        return en, pu

    def build_dict(self, sentences, max_words=50000):
        """
        传入load_data构造的分词后的列表数据
        构建词典(key为单词，value为id值)
        """
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    def wordToID(self, en, pu, en_dict, pu_dict, sort=True):
        """
        将翻译前(英文)数据和翻译后(旁遮普)数据的单词列表表示的数据
        均转为id列表表示的数据
        """
        length = len(en)
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_pu_ids = [[pu_dict.get(w, 0) for w in sent] for sent in pu]

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_pu_ids = [out_pu_ids[i] for i in sorted_index]

        return out_en_ids, out_pu_ids

    def splitBatch(self, en, pu, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_pu = [pu[index] for index in batch_index]
            batch_pu = seq_padding(batch_pu)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_pu))

        return batches

    def save_to_file(self):
        data_list = [self.pu_word_dict, self.en_word_dict, self.pu_index_dict, self.en_index_dict]
        file_name_list = ["pu_word_dict", "en_word_dict", "pu_index_dict", "en_index_dict"]
        for i, data in enumerate(data_list):
            with open('data/word_name_dict/' + file_name_list[i] + '.csv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for k, v in data.items():
                    writer.writerow([k, v])

        print('preparing data .....')

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run():
    from setting import TRAIN_FILE, DEV_FILE
    PrepareData(TRAIN_FILE, DEV_FILE).save_to_file()

if __name__ == '__main__':
    run()


