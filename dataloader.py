import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import utils_data_processing_1 as up
import pickle
import pandas as pd

train_data_path = './Processed_data/train.csv'
train_text_path = "./captions/training_label.json"
train_feature_dir = "./feature_dirs_training"
train_path = [train_data_path, train_text_path, train_feature_dir]

test_data_path = './Processed_data/test.csv'
test_text_path = "./captions/testing_label.json"
test_feature_dir = "./feature_dirs_testing"
test_path = [test_data_path, test_text_path, test_feature_dir]

with open('./Processed_data/word2idx.pkl', 'rb') as word2idx_file:
    word2idx = pickle.load(word2idx_file)

with open('./Processed_data/idx2word.pkl', 'rb') as idx2word_file:
    idx2word = pickle.load(idx2word_file)


class VideoDataset(Dataset):
    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode

        if self.mode == "train":
            self.path = train_path
        else:
            self.path = test_path
        self.data = pd.read_csv(self.path[0])

        self.word2idx = word2idx
        self.idx2word = idx2word

        self.max_len = 11

    def __getitem__(self, idx):
        var = self.data.loc[idx]
        var_data = self.data.loc[[idx]]
        video_path = var["video_path"]
        video_caption = var["caption"]
        video_id = var["id"]
        current_feature_value = np.load(video_path)
        current_captions = up.get_captions_list_sampling_1(var_data)
        current_captions[0] = current_captions[0] + ' <eos>'
        s_vector = [word2idx[x.lower()] for x in current_captions[0].split(' ')]
        s_vector = np.array(s_vector)
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        label = np.hstack((s_vector, np.zeros(self.max_len - len(s_vector))))
        non_zero = (label != 0).nonzero()
        mask[:len(non_zero[0])] = 1

        data = {}
        data["fc_feats"] = torch.from_numpy(current_feature_value).type(torch.FloatTensor)
        data["label"] = torch.from_numpy(label).type(torch.LongTensor)
        data["mask"] = torch.from_numpy(mask).type(torch.FloatTensor)
        data["id"] = video_id
        return data

    def __len__(self):
        return len(self.data)



