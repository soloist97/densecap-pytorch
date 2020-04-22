import os
import json

import torch
import numpy as np
from torch.utils.data.dataset import Subset
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import DenseCapDataset, DataLoaderPFG
from model.densecap import densecap_resnet50_fpn

from evaluate import quality_check, quantity_check

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 30
USE_TB = False
CONFIG_PATH = './model_params'
MODEL_NAME = 'debug'
IMG_DIR_ROOT = './data/visual-genome'
VG_DATA_PATH = './data/VG-regions.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts.pkl'
MAX_TRAIN_IMAGE = 10  # if -1, use all images in train set
MAX_VAL_IMAGE = 10


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
    args['fusion_type'] = 'merge'

    # Training Settings
    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-6
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0.
    args['batch_size'] = 2
    args['use_pretrain_fasterrcnn'] = True

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f)

    return args


def save_model(model, results_on_val):

    state = {'model': model.state_dict(),
             'results_on_val':results_on_val}
    filename = os.path.join('model_params', '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


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
    if args['use_pretrain_fasterrcnn']:
        model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)

    model.to(device)

    optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                    if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': args['caption_lr']}],
                                  lr=args['lr'], weight_decay=args['weight_decay'])

    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    iter_counter = 0
    best_map = 0.

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

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

            # record loss
            if USE_TB:
                writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)

                writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)


            if iter_counter % (MAX_TRAIN_IMAGE/(args['batch_size']*4)) == 0:
                print("[{}][{}]\ntotal_loss {:.3f}".format(epoch, batch, total_loss.item()))
                for k, v in losses.items():
                    print(" <{}> {:.3f}".format(k, v))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iter_counter += 1

        try:
            results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
            if results['map'] > best_map:
                best_map = results['map']
                save_model(model, results)

            if USE_TB:
                writer.add_scalar('metric/map', results['map'], iter_counter)
                writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

        except AssertionError as e:
            print('[INFO]: evaluation failed at epoch {}'.format(epoch))
            print(e)

    if USE_TB:
        writer.close()


if __name__ == '__main__':

    args = set_args()
    train(args)
