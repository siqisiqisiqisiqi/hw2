import json
import os

import numpy as np
import torch
import torch.optim as optim
from dataloader import VideoDataset
from S2VTModel import S2VTModel
from torch.utils.data import DataLoader
import misc.utils as utils
from torch.nn.utils import clip_grad_value_
import pandas as pd
import ATT_S2VTModel as att

batch_size = 64
vocab_size = 1066
max_len = 11
dim_hidden = 512
dim_vid = 4096
dim_word = 512

learning_rate = 4e-4    # 4e-4
weight_decay = 5e-4     # 5e-4
learning_rate_decay_step_size = 200
learning_rate_decay_rate = 0.8
epochs = 201
gradient_clip = 5

save_every_epoch = 50
model_folder_path = './model/'

schedule_sample_decay_par = 0.97


def train(dataloader, model, loss_fn, optimizer, lr_scheduler):
    model.train()
    loss_dic = []
    for epoch in range(epochs):
        iteration = 0
        ss_decay = np.power(schedule_sample_decay_par, epoch)
        for data in dataloader:
            fc_feats = data["fc_feats"]
            labels = data["label"]
            masks = data["mask"]
            if np.random.rand(1)[0] < ss_decay:
                mode = 'train'
            else:
                mode = 'inference'
            seq_probs, _ = model(fc_feats, labels, mode)
            loss = loss_fn(seq_probs, labels[:, 1:], masks[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            clip_grad_value_(model.parameters(), gradient_clip)
            optimizer.step()
            train_loss = loss.item()
            if iteration % 5 == 0:
                loss_dic.append(train_loss)

            iteration += 1
            print("iter %d (epoch %d), train_loss = %.6f" %
                  (iteration, epoch, train_loss))

        if epoch % save_every_epoch == 0:
            model_path = os.path.join(model_folder_path,
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(model_folder_path,
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))
        lr_scheduler.step()
    col = ["loss"]
    df = pd.DataFrame(loss_dic, columns=col)
    df.to_csv('./model/test.csv', index=False)



def main(model_type):
    dataset = VideoDataset("train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

    loss_fn = utils.LanguageModelCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=learning_rate_decay_step_size,
        gamma=learning_rate_decay_rate)

    train(dataloader, model, loss_fn, optimizer, exp_lr_scheduler)


if __name__ == '__main__':
    main('ATT_S2VT')



