
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
from scipy import random
from sklearn import preprocessing
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from retrieval_model import FOP

def read_data(ver, train_lang):
    train_file_face = f"./pre_extracted_features/{ver}/{train_lang}/{train_lang}_faces_train.csv"
    
    if FLAGS.ge2e_voice:
        train_file_voice = f"./GE2E_Embeddings/mavceleb_{ver}_train/{train_lang}.csv"
    else:    
        train_file_voice = f"./pre_extracted_features/{ver}/{train_lang}/{train_lang}_voices_train.csv"
    
    print('Reading Train Faces')
    img_train = pd.read_csv(train_file_face, header=None)
    img_train = np.asarray(img_train)
    train_label = img_train[:, -1]
    img_train = img_train[:, :-1]
    print('Reading Voices')
    voice_train = pd.read_csv(train_file_voice, header=None, low_memory=False)
    voice_train = np.asarray(voice_train)
    if FLAGS.ge2e_voice:
        voice_train = voice_train[1: , 4:].astype(np.float)
    else:
        voice_train = voice_train[:, :-1].astype(np.float)
    
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    print("Train file length", len(img_train))
        
    print('Shuffling\n')
    combined = list(zip(img_train, voice_train, train_label))
    img_train = []
    voice_train = []
    train_label = []
    random.shuffle(combined)
    img_train[:], voice_train, train_label[:] = zip(*combined)
    combined = [] 
    img_train = np.asarray(img_train).astype(np.float)
    voice_train = np.asarray(voice_train).astype(np.float)
    train_label = np.asarray(train_label)
    
    
    return img_train, voice_train, train_label
 
def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(ver, train_lang, face_train, voice_train, train_label):
    
    n_class = 64 if ver == 'v1' else 78
    model = FOP(FLAGS.cuda, FLAGS.fusion, FLAGS.dim_embed, FLAGS.mid_att_dim, face_train.shape[1], voice_train.shape[1], n_class, FLAGS.num_layers)
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss().cuda()
    opl_loss = OrthogonalProjectionLoss().cuda()
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True
    
    # =============================================================================
    #     For Linear Fusion
    # =============================================================================
    
    if FLAGS.fusion == 'linear':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                        {'params' : model.fusion_layer.weight1},
                        {'params' : model.fusion_layer.weight2}]
    
    
    # =============================================================================
    #     For Gated Fusion
    # =============================================================================
    
    elif FLAGS.fusion == 'gated':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                      {'params' : model.fusion_layer.attention.parameters()}]
        
    # =============================================================================
    #     For MultiGated Fusion
    # =============================================================================
    
    elif FLAGS.fusion == 'multigated':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()}]
        # Include parameters of attention layers
        for attention_layer in model.fusion_layer.attention_layers:
            parameters.append({'params': attention_layer.parameters()})

    optimizer = optim.Adam(parameters, lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f"  + Number of params: {n_parameters}")
    
    epoch=1
    num_of_batches = (len(train_label) // FLAGS.batch_size)
    
    
    save_dir = f"./models/{ver}/{train_lang}/{'GE2E_voice_' if FLAGS.ge2e_voice else ''}{ver}_{train_lang}_{FLAGS.fusion}_alpha_{FLAGS.alpha:0.2f}"
    best_model_dir = f"./models/{ver}/{train_lang}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_epoch_loss = float('inf')

    while (epoch <= FLAGS.epochs):
        loss_per_epoch = 0
        loss_plot = []
        print(f"Epoch {epoch: 03d}")
        for idx in tqdm(range(num_of_batches)):
            face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
            voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
            loss_tmp, loss_opl, loss_soft, s_fac, d_fac = train(face_feats, voice_feats, 
                                                         batch_labels, 
                                                         model, optimizer, ce_loss, opl_loss, FLAGS.alpha)
            loss_per_epoch+=loss_tmp
        
        loss_per_epoch/=num_of_batches
        
        loss_plot.append(loss_per_epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict()}, save_dir, f"checkpoint_{epoch: 03d}.pth.tar")

        print(f"==> Epoch: {epoch}/{FLAGS.epochs} Loss: {loss_per_epoch: 0.2f} Alpha: {FLAGS.alpha: 0.2f} ")
        
        if epoch > 1:
            if (loss_per_epoch - best_epoch_loss) / best_epoch_loss > FLAGS.early_stop_criterion:
                print(f"{'----- EARLY STOPPING -----': ^30}")
                return
            if loss_per_epoch < best_epoch_loss:    
                best_epoch_loss = loss_per_epoch
                save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()}, best_model_dir, f"best_checkpoint{'_GE2E_voice' if FLAGS.ge2e_voice else ''}.pth.tar")
                print(f"{'+++++ BEST MODEL SO FAR +++++': ^30}")
            
        loss_per_epoch = 0
        epoch += 1
            
    return
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = 1.0 - pos_pairs_mean + neg_pairs_mean

        return loss, pos_pairs_mean, neg_pairs_mean


def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, alpha):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    face_feats = torch.from_numpy(face_feats).float()
    voice_feats = torch.from_numpy(voice_feats).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

    face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)
    comb, face_embeds, voice_embeds = model.train_forward(face_feats, voice_feats, labels)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)
    
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average
 
def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--version', default='v1', type=str, help='Dataset version')
    parser.add_argument('--train_lang', default='Urdu', type=str, help='Training language')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs to train, number')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha Value')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    parser.add_argument('--train_all_langs', action='store_true', default=False, help='Training all possible language combinations')
    parser.add_argument('--mid_att_dim', type=int, default=128,
                        help='Used only in case of gated fusion, it is Intermediate Embedding Size (Inside Attention Algorithm)')
    parser.add_argument('--early_stop_criterion', type=float, default=1e-3,
                        help='Minimum relative epoch loss improvement')
    parser.add_argument('--ge2e_voice', action='store_true', default=False, help='Uses GE2E precomputed voice embeddings')
    parser.add_argument('--num_layers', type=int, default=3, help='Used only in case of multigated fusion,')
    
    
    print('Training')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    
    if FLAGS.train_all_langs:
        vers = ['v1', 'v1', 'v2', 'v2']
        train_langs = ['English', 'Urdu', 'English', 'Hindi']
    else:
        vers = [FLAGS.version]
        train_langs = [FLAGS.train_lang]

    for i in range(len(vers)):
        ver = vers[i]
        train_lang = train_langs[i]

        print("="*30)
        print(f"Version of the Dataset: {ver}")
        print(f"Training Language: {train_lang}")
        print("-"*30)

        face_train, voice_train, train_label = read_data(ver, train_lang)
        main(ver, train_lang, face_train, voice_train, train_label)

        print("="*30)
