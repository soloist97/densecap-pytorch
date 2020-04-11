import os
import json

import torch
import numpy as np
from torch.utils.data.dataset import Subset

from utils.data_loader import DenseCapDataset, DataLoaderPFG
from model.densecap import densecap_resnet50_fpn


torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 100
CONFIG_PATH = './model_params'
MODEL_NAME = 'debug'
IMG_DIR_ROOT = './data/visual-genome'
VG_DATA_PATH = './data/VG-regions.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts.pkl'
MAX_TRAIN_IMAGE = 10  # if -1, use all images in train set


def set_args():

    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False,

    # Caption parameters
    args['feat_size'] = 4096
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['rnn_num_layers'] = 1
    args['vocab_size'] = 10629
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 0.1
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-3
    args['batch_size'] = 4

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f)

    return args


def train(args):

    print('Model {} start training...'.format(MODEL_NAME))

    model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                                  feat_size=args['feat_size'],
                                  hidden_size=args['hidden_size'],
                                  max_len=args['max_len'],
                                  emb_size=args['emb_size'],
                                  rnn_num_layers=args['rnn_num_layers'],
                                  vocab_size=args['vocab_size'],
                                  fusion_type=args['fusion_type'])

    model.to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args['lr'])

    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))

    train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    for epoch in range(MAX_EPOCHS):

        for batch, (img, targets, info) in enumerate(train_loader):

            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k:v.to(device) for k, v in target.items()} for target in targets]

            model.train()
            losses = model(img, targets)

            detect_loss =  losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                           losses['loss_classifier'] + losses['loss_box_reg']
            caption_loss = losses['loss_caption']


            total_loss = args['detect_loss_weight'] * detect_loss + args['caption_loss_weight'] * caption_loss
            print('[{}][{}] total_loss {}'.format(epoch, batch, total_loss.item()))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

if __name__ == '__main__':

    args = set_args()
    train(args)

