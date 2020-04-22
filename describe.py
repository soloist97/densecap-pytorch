import os
import json
import pickle
import argparse

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from model.densecap import densecap_resnet50_fpn


def load_model(console_args):

    with open(console_args.config_json, 'r') as f:
        model_args = json.load(f)

    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  rnn_num_layers=model_args['rnn_num_layers'],
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'])

    checkpoint = torch.load(console_args.model_checkpoint)
    model.load_state_dict(checkpoint['model'])

    if console_args.verbose and 'results_on_val' in checkpoint.keys():
        print('[INFO]: checkpoint {} loaded'.format(console_args.model_checkpoint))
        print('[INFO]: correspond performance on val set:')
        for k, v in checkpoint['results_on_val'].items():
            if not isinstance(v, dict):
                print('        {}: {:.3f}'.format(k, v))

    return model


def get_image_path(console_args):

    img_list = []

    if os.path.isdir(console_args.img_path):
        for file_name in os.listdir(console_args.img_path):
            img_list.append(os.path.join(console_args.img_path, file_name))
    else:
        img_list.append(console_args.img_path)

    return img_list


def img_to_tensor(img_list):

    assert isinstance(img_list, list) and len(img_list) > 0

    img_tensors = []

    for img_path in img_list:

        img = Image.open(img_path).convert("RGB")

        img_tensors.append(transforms.ToTensor()(img))

    return img_tensors


def describe_images(model, img_list, device, console_args):

    assert isinstance(img_list, list)
    assert isinstance(console_args.batch_size, int) and console_args.batch_size > 0

    all_results = []

    with torch.no_grad():

        model.to(device)
        model.eval()
        model.return_features = False

        for i in tqdm(range(0, len(img_list), console_args.batch_size), disable=not console_args.verbose):

            image_tensors = img_to_tensor(img_list[i:i+console_args.batch_size])
            input_ = [t.to(device) for t in image_tensors]

            results = model(input_)

            all_results.extend(results)

    return all_results


def save_results_to_file(img_list, all_results, console_args):

    with open(os.path.join(console_args.lut_path), 'rb') as f:
        look_up_tables = pickle.load(f)

    idx_to_token = look_up_tables['idx_to_token']

    results_dict = {}
    for img_path, results in zip(img_list, all_results):

        if console_args.verbose:
            print('[Result] ==== {} ====='.format(img_path))

        results_dict[img_path] = []
        for box, cap, score in zip(results['boxes'], results['caps'], results['scores']):

            r = {
                'box': [round(c, 2) for c in box.tolist()],
                'score': round(score.item(), 2),
                'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                                if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
            }

            if console_args.verbose and r['score'] > 0.8:
                print('        SCORE {}  BOX {}'.format(r['score'], r['box']))
                print('        CAP {}\n'.format(r['cap']))

            results_dict[img_path].append(r)

    if not os.path.exists(console_args.result_dir):
        os.mkdir(console_args.result_dir)
    with open(os.path.join(console_args.result_dir, 'result.json'), 'w') as f:
        json.dump(results_dict, f)

    if console_args.verbose:
        print('[INFO] result save to {}'.format(os.path.join(console_args.result_dir, 'result.json')))


def main(console_args):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # === prepare images ====
    img_list = get_image_path(console_args)

    # === prepare model ====
    model = load_model(console_args)

    # === inference ====
    all_results = describe_images(model, img_list, device, console_args)

    # === save results ====
    save_results_to_file(img_list, all_results, console_args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do dense captioning')
    parser.add_argument('--config_json', type=str, help="path of the json file which stored model configuration")
    parser.add_argument('--lut_path', type=str, default='./data/VG-regions-dicts.pkl', help='look up table path')
    parser.add_argument('--model_checkpoint', type=str, help="path of the trained model checkpoint")
    parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images")
    parser.add_argument('--result_dir', type=str, default='.',
                        help="path of the directory to save the output file")
    parser.add_argument('--batch_size', type=int, default=1, help="useful when img_path is a directory")
    parser.add_argument('--cpu', action='store_true', help='whether use cpu to compute')
    parser.add_argument('--verbose', action='store_true', help='whether output info')
    args = parser.parse_args()

    main(args)
