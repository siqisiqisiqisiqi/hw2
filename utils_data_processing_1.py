import pandas as pd
import os
import numpy as np
import pickle
import sys

def get_data_1(text_path, feat_path):
    data = pd.read_json(text_path)
    for i in range(len(data)):
        data["caption"][i] = get_shortest_caption(data["caption"][i])
        # data["caption"][i] = list(map(lambda x: x.replace('.', ''), data["caption"][i]))
        # print(data["caption"][i])
    data['video_path'] = data['id'].map(lambda x: x + ".npy")
    data['video_path'] = data['video_path'].map(lambda x: os.path.join(feat_path, x))
    data = data[data['video_path'].map(lambda x: os.path.exists(x))]
    data = data[data["caption"].map(lambda x: isinstance(x, str))]
    unique_filenames = data['video_path'].unique()
    data = data[data['video_path'].map(lambda x: x in unique_filenames)]
    return data


def get_shortest_caption(list_1):
    length = 100
    caption_s = ""
    for list_s in list_1:
        if length > len(list_s):
            caption_s = list_s
            length = len(list_s)
    return caption_s


def create_word_dict_1(sentence_iterator, word_count_threshold=5):
    word_counts = {}
    sent_cnt = 0

    for sent in sentence_iterator:
        sent_cnt += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    idx2word = {}
    idx2word[0] = '<pad>'
    idx2word[1] = '<bos>'
    idx2word[2] = '<eos>'
    idx2word[3] = '<unk>'

    word2idx = {}
    word2idx['<pad>'] = 0
    word2idx['<bos>'] = 1
    word2idx['<eos>'] = 2
    word2idx['<unk>'] = 3

    for idx, w in enumerate(vocab):
        word2idx[w] = idx + 4
        idx2word[idx + 4] = w

    word_counts['<pad>'] = sent_cnt
    word_counts['<bos>'] = sent_cnt
    word_counts['<eos>'] = sent_cnt
    word_counts['<unk>'] = sent_cnt

    return word2idx, idx2word


def get_captions_list_1(data):
    train_captions = data['caption'].values

    captions_list = list(train_captions)
    captions = np.asarray(captions_list, dtype=object)

    captions = list(map(lambda x: x.replace('.', ''), captions))
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))

    return captions


def get_captions_list_sampling_1(data):
    current_captions = data['caption'].values
    current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
    current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
    current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('/', ''), current_captions))

    return current_captions


if __name__ == '__main__':

    test_feat_path = sys.argv[1]
    test_feature_dir = os.path.join(test_feat_path, "feat")
    test_text_path = os.path.join(test_feat_path, "testing_label.json")
    # print(test_feat_path)
    # print(test_text_path)
    # train_text_path = "./captions/training_label.json"
    # train_feature_dir = "./feature_dirs_training"
    # test_text_path = "./captions/testing_label.json"
    # test_feature_dir = "./feature_dirs_testing"

    # train_data = get_data_1(train_text_path, train_feature_dir)
    test_data = get_data_1(test_text_path, test_feature_dir)
    # train_data.to_csv('./Processed_data/train.csv', index=False)
    test_data.to_csv('./Processed_data/test.csv', index=False)
    # captions = get_captions_list_1(train_data)
    # print(train_data["id"])
    # test_captions = get_captions_list_1(test_data)
    # captions = captions + test_captions
    # # print(len(captions))
    #


    # max_len = 0
    # for i in range(len(captions)):
    #     caption = captions[i]
    #     num = len(caption.split())
    #     if num > max_len:
    #         max_len = num


    # word2idx, idx2word = create_word_dict_1(captions, word_count_threshold=0)
    # print(len(idx2word))
    #
    #
    # with open("./Processed_data/word2idx.pkl", 'wb') as word2idx_file:
    #     pickle.dump(word2idx, word2idx_file, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open("./Processed_data/idx2word.pkl", 'wb') as idx2word_file:
    #     pickle.dump(idx2word, idx2word_file, protocol=pickle.HIGHEST_PROTOCOL)
