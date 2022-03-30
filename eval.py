import json
import os
import sys
import numpy as np
import pickle
import torch
import torch.optim as optim
from dataloader import VideoDataset
from S2VTModel import S2VTModel
from torch import nn
from torch.utils.data import DataLoader
import ATT_S2VTModel as att

batch_size = 64
vocab_size = 1066
max_len = 11
dim_hidden = 512
dim_vid = 4096
dim_word = 512

model_path = './model/model_200.pth'
# caption_test_path = 'generated_video_caption.txt'

with open('./Processed_data/word2idx.pkl', 'rb') as word2idx_file:
    word2idx = pickle.load(word2idx_file)

with open('./Processed_data/idx2word.pkl', 'rb') as idx2word_file:
    idx2word = pickle.load(idx2word_file)



def test(model, dataset):

    caption_test_path = sys.argv[1]
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    sentence_dic = []
    video_id_dic = []
    for data in loader:

        fc_feats = data["fc_feats"]
        labels = data["label"]
        video_id = data["id"]
        video_id_dic = video_id_dic + video_id

        with torch.no_grad():
            seq_probs, seq_preds = model(fc_feats, mode="inference")

        preds_id = seq_preds.tolist()
        for key in preds_id:
            words = [idx2word[x] for x in key]
            punct = np.argmax(np.array(words) == '<eos>')
            words = words[:punct]
            sentence = ' '.join(words)
            sentence_dic.append(sentence)

    with open(caption_test_path, mode="w+", encoding="utf-8") as file:
        for i in range(len(sentence_dic)):
            file.write(video_id_dic[i]+',')
            file.write(sentence_dic[i]+'\n')


def main(model_type):
    dataset = VideoDataset("test")
    if model_type == 'S2VT':
        model = S2VTModel(
            vocab_size,
            max_len,
            dim_hidden,
            dim_vid,
            dim_word
        )
    else:
        encoder = att.EncoderRNN(
            dim_vid,
            dim_hidden
        )
        decoder = att.DecoderRNN(
            vocab_size,
            max_len,
            dim_hidden,
            dim_word
        )
        model = att.S2VTAttModel(encoder, decoder)
    model.load_state_dict(torch.load(model_path))
    test(model, dataset)


if __name__ == "__main__":

    main('ATTS2VT')


