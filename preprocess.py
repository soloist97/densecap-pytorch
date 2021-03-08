# coding=utf8
import re, argparse, json, string, pickle
from collections import Counter


import h5py
import numpy as np


"""
Modified from https://github.com/jcjohnson/densecap/blob/master/preprocess.py

Format in region_descriptions.json
{
  "id": [int], Unique identifier for this image,
  "regions": [
    {
      "id": [int] Unique identifier for this region,
      "image": [int] ID of the image to which this region belongs,
      "height": [int] Height of the region in pixels,
      "width": [int] Width of the region in pixels,
      "phrase": [string] Caption for this region,
      "x": [int] x-coordinate of the upper-left corner of the region,
      "y": [int] y-coordinate of the upper-left corner of the region,
    },
    ...
  ]
}

Format in image_data.json
  {
    "image_id": [int] Unique identifier for this image,
    "url": [str] Visual Genome-hosted image URL
    "width": [int] Width of the image in pixels,
    "height": [int] Height of the image in pixels,
    "coco_id": [int] ID of the image in the coco dataset
    "flickr_id": [int] ID of the image in the flickr dataset
  }


We assume that all images are on disk in a two folder (VG_100K and VG_100K_2), and that
the filename for each image is the same as its id with a .jpg extension.

This file will be preprocessed into an HDF5 file and a Pickle file with
some auxiliary information. The captions will be tokenized with some
basic preprocessing (split by words, remove special characters).

Note, in general any indices anywhere in input/output of this file are 0-indexed.

The output Pickle file is an object with the following elements:
- token_to_idx: Dictionary mapping strings to integers for encoding tokens, 
                in 1-indexed format.
- filename_to_idx: Dictionary mapping string filenames to indices.
- idx_to_token: Inverse of the above.
- idx_to_filename: Inverse of the above.
- idx_to_directory: Dictionary mapping indices to directory [str]('VG_100K' or 'VG_100K_2').
- split: Split of train, test, val {'train': [idx, ...], 'val': [...], 'test': [...]}

The output HDF5 file has the following format to describe N images with
M total regions:

- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (xc, yc, w, h) where yc and xc are center coordinates of the box,
  and are one-indexed.
- lengths: int32 array of shape (M,) giving lengths of label sequence for each box
- captions: int32 array of shape (M, L) giving the captions for each region.
  Captions in the input with more than L = --max_token_length tokens are
  discarded. To recover a token from an integer in this matrix,
  use idx_to_token from the Pickle output file. Padded with zeros.
- img_to_first_box: int32 array of shape (N,). If img_to_first_box[i] = j then
  captions[j] and boxes[j] give the first annotation for image i
  (using one-indexing).
- img_to_last_box: int32 array of shape (N,). If img_to_last_box[i] = j then
  captions[j] and boxes[j] give the last annotation for image i
  (using one-indexing).
- box_to_img: int32 array of shape (M,). If box_to_img[i] = j then then
  regions[i] and captions[i] refer to images[j] (using one-indexing).
"""


def build_vocab(data, min_token_instances, verbose=True):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    token_counter = Counter()
    for img in data:
        for region in img['regions']:
            if region['tokens'] is not None:
                token_counter.update(region['tokens'])
    vocab = set()
    for token, count in token_counter.items():
        if count >= min_token_instances:
            vocab.add(token)

    if verbose:
        print('Keeping {} / {} tokens with enough instances'.format(len(vocab), len(token_counter)))

    vocab = list(vocab)
    vocab = sorted(vocab, key=lambda token: token_counter[token], reverse=True)
    if len(vocab) < len(token_counter):
        vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab
        if verbose:
            print('adding special <pad> <bos> <eos> <unk> token.')
    else:
        vocab = ['<pad>', '<bos>', '<eos>'] + vocab
        if verbose:
            print('adding special <pad> <bos> <eos> token.')

    return vocab


def build_vocab_dict(vocab):
    token_to_idx, idx_to_token = {}, {}
    next_idx = 0  # 0-indexed

    for token in vocab:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def encode_caption(tokens, token_to_idx, max_token_length):
    encoded = np.ones(max_token_length+2, dtype=np.int64) * token_to_idx['<pad>']
    encoded[0] = token_to_idx['<bos>']
    encoded[len(tokens)+1] = token_to_idx['<eos>']

    for i, token in enumerate(tokens):

        if token in token_to_idx:
            encoded[i+1] = token_to_idx[token]
        else:
            encoded[i+1] = token_to_idx['<unk>']

    return encoded


def encode_captions(data, token_to_idx, max_token_length):
    encoded_list = []
    lengths = []
    for img in data:
        for region in img['regions']:
            tokens = region['tokens']
            if tokens is None: continue
            tokens_encoded = encode_caption(tokens, token_to_idx, max_token_length)
            encoded_list.append(tokens_encoded)
            lengths.append(len(tokens)+2)
    return np.vstack(encoded_list), np.asarray(lengths, dtype=np.int64)  # in pytorch np.int64 is torch.long


def encode_boxes(data, image_data, all_image_ids):
    all_boxes = []
    for i, img in enumerate(data):

        img_info = image_data[all_image_ids.index(img['id'])]
        assert img['id'] == img_info['image_id'], 'id mismatch'

        for region in img['regions']:
            if region['tokens'] is None:
                continue

            x1, y1 = region['x'], region['y']
            x2, y2 = x1 + region['width'], y1 + region['height']

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0

            # sanity check
            try:
                assert x1 <= img_info['width'], 'invalid x1 coordinate {} > {} in image_id:{} box_id:{}'.format(x1, img_info['width'], img['id'], region['region_id'])
                assert y1 <= img_info['height'], 'invalid y1 coordinate {} > {} in image_id:{} box_id:{}'.format(y1, img_info['height'], img['id'], region['region_id'])
                assert x2 <= img_info['width'], 'invalid x2 coordinate {} > {} in image_id:{} box_id:{}'.format(x2, img_info['width'], img['id'], region['region_id'])
                assert y2 <= img_info['height'], 'invalid y2 coordinate {} > {} in image_id:{} box_id:{}'.format(y2, img_info['height'], img['id'], region['region_id'])
            except AssertionError as e:
                print(e)

                print('orignal bbox coordinate ', (x1, y1, x2, y2))
                # clamp to image
                if x1 > img_info['width']: x1 = (img_info['width']- 1) - region['width']
                if y1 > img_info['height']: y1 = (img_info['height'] - 1) - region['height']
                if x2 > img_info['width']: x2 = img_info['width']- 1
                if y2 > img_info['height']: y2 = img_info['height'] - 1
                print('clamped bbox coordinate ', (x1, y1, x2, y2))

            box = np.asarray([x1, y1, x2, y2], dtype=np.int32)
            all_boxes.append(box)
    return np.vstack(all_boxes)


def build_img_idx_to_box_idxs(data):
    img_idx = 0
    box_idx = 0
    num_images = len(data)
    img_to_first_box = np.zeros(num_images, dtype=np.int32)
    img_to_last_box = np.zeros(num_images, dtype=np.int32)
    for img in data:
        img_to_first_box[img_idx] = box_idx
        for region in img['regions']:
            if region['tokens'] is None: continue
            box_idx += 1
        img_to_last_box[img_idx] = box_idx - 1  # -1 to make these inclusive limits 闭集
        img_idx += 1

    return img_to_first_box, img_to_last_box


def build_filename_dict(data):
    # First make sure all filenames
    filenames_list = ['%d.jpg' % img['id'] for img in data]
    assert len(filenames_list) == len(set(filenames_list))

    next_idx = 0
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = '%d.jpg' % img['id']
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename


def build_directory_dict(data, image_data, all_image_ids):

    idx_to_directory = dict()

    next_idx = 0
    for img in data:

        img_info = image_data[all_image_ids.index(img['id'])]
        assert img['id'] == img_info['image_id'], 'id mismatch'

        idx_to_directory[next_idx] = re.search('(VG.*)/(.*.jpg)$', img_info['url']).group(1)
        next_idx += 1

    return idx_to_directory


def encode_filenames(data, filename_to_idx):
    filename_idxs = []
    for img in data:
        filename = '%d.jpg' % img['id']
        idx = filename_to_idx[filename]
        for region in img['regions']:
            if region['tokens'] is None: continue
            filename_idxs.append(idx)
    return np.asarray(filename_idxs, dtype=np.int32)


def words_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    translator = str.maketrans('', '', string.punctuation)
    replacements = {
        u'½': u'half',
        u'—': u'-',
        u'™': u'',
        u'¢': u'cent',
        u'ç': u'c',
        u'û': u'u',
        u'é': u'e',
        u'°': u' degree',
        u'è': u'e',
        u'…': u'',
    }

    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(translator).split()


def split_filter_captions(data, max_token_length, tokens_type, verbose=True):
    """
    Modifies data in-place by adding a 'tokens' field to each region.
    If the region's label is too long, 'tokens' will be None; otherwise
    it will be a list of strings.
    Splits by space when tokens_type = "words", or lists all chars when "chars"
    """
    captions_kept = 0
    captions_removed = 0
    for i, img in enumerate(data):
        if verbose and (i + 1) % 2000 == 0:
            print('Splitting tokens in image {} / {}'.format(i + 1, len(data)))
        img_kept, img_removed = 0, 0
        for region in img['regions']:

            # create tokens array
            if tokens_type == 'words':
                tokens = words_preprocess(region['phrase'])  # 此处分词
            elif tokens_type == 'chars':
                tokens = list(region['label'])
            else:
                assert False, 'tokens_type must be "words" or "chars"'

            # filter by length
            if max_token_length > 0 and len(tokens) <= max_token_length:  # 过长的描述被丢弃而非截断
                region['tokens'] = tokens
                captions_kept += 1
                img_kept += 1
            else:
                region['tokens'] = None
                captions_removed += 1
                img_removed += 1

        if img_kept == 0:
            print('kept {}, removed {}'.format(img_kept, img_removed))
            assert False, 'DANGER, some image has no valid regions. Not super sure this doesnt cause bugs. Think about more if it comes up'

    if verbose:
        print('Keeping {} captions'.format(captions_kept))
        print('Skipped {} captions for being too long'.format(captions_removed))


def encode_splits(data, split_data):
    """ Encode splits by mappings """
    id_to_split = {}
    for split, idxs in split_data.items():
        for idx in idxs:
            id_to_split[idx] = split

    split_dict = {k:list() for k in split_data.keys()}

    for i, img in enumerate(data):
        split_dict[id_to_split[img['id']]].append(i)

    return split_dict


def filter_images(data, split_data):
    """ Keep only images that are in some split and have some captions """
    all_split_ids = set()
    for split_name, ids in split_data.items():
        all_split_ids.update(ids)
    new_data = []
    for img in data:
        keep = img['id'] in all_split_ids and len(img['regions']) > 0
        if keep:
            new_data.append(img)
    return new_data


def main(args):
    # read in the data
    with open(args.region_data, 'r') as f:
        data = json.load(f)
    with open(args.split_json, 'r') as f:
        split_data = json.load(f)  # {'train':[...], 'test':[...], 'val':[...]}
    with open(args.image_data, 'r') as f:
        image_data = json.load(f)

    all_image_ids = [image_data[i]['image_id'] for i in range(len(image_data))]

    # Only keep images that are in a split
    print(f'There are {len(data)} images total')
    data = filter_images(data, split_data)  # data为vg数据集定义的region description的格式的列表
    print(f'After filtering for splits there are {len(data)} images')

    if args.max_images > 0:
        data = data[:args.max_images]

    # add split information
    split = encode_splits(data, split_data)  # dict {'train': [idx, ...], 'val': [...], 'test': [...]}

    # create the output hdf5 file handle
    with h5py.File(args.h5_output, 'w') as f:

        # process "label" field in each region to a "tokens" field, and cap at some max length
        # 即在data中每一个region处增加一个token键以及分词后的序列 长于max_token_length的描述被丢弃
        split_filter_captions(data, args.max_token_length, args.tokens_type)

        # build vocabulary
        vocab = build_vocab(data, args.min_token_instances)  # vocab is a list()
        token_to_idx, idx_to_token = build_vocab_dict(vocab)  # both mappings are dicts 且从0开始计数

        # encode labels
        # captions_matrix (M, max_token_length) 其中M为region总数
        # lengths_vector (M, )
        captions_matrix, lengths_vector = encode_captions(data, token_to_idx, args.max_token_length)
        f.create_dataset('captions', data=captions_matrix)
        f.create_dataset('lengths', data=lengths_vector)

        # encode boxes
        # 相对image_size拉伸标签中的bbox坐标参数 (M, 4)
        boxes_matrix = encode_boxes(data, image_data, all_image_ids)
        f.create_dataset('boxes', data=boxes_matrix)

        # integer mapping between image ids and box ids
        # img_idx是所有图像在data中的顺序 为了方便访问bbox 记录在M个region里起始和结束idx
        img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
        f.create_dataset('img_to_first_box', data=img_to_first_box)
        f.create_dataset('img_to_last_box', data=img_to_last_box)
        # 建立img_idx与文件名的映射关系
        filename_to_idx, idx_to_filename = build_filename_dict(data)
        idx_to_directory = build_directory_dict(data, image_data, all_image_ids)
        box_to_img = encode_filenames(data, filename_to_idx)
        f.create_dataset('box_to_img', data=box_to_img)

    # and write the additional pickle file
    pickle_struct = {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
        'filename_to_idx': filename_to_idx,
        'idx_to_filename': idx_to_filename,
        'idx_to_directory': idx_to_directory,
        'split': split,
    }
    with open(args.pickle_output, 'wb') as f:
        pickle.dump(pickle_struct, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # INPUT settings
    parser.add_argument('--region_data',
                        default='data/visual-genome/region_descriptions.json',
                        help='Input JSON file with regions and captions')
    parser.add_argument('--image_data',
                        default='data/visual-genome/image_data.json',
                        help='Input JSON file with image url weight and height')
    parser.add_argument('--split_json',
                        default='info/densecap_splits.json',
                        help='JSON file of splits')

    # OUTPUT settings
    parser.add_argument('--pickle_output',
                        default='data/VG-regions-dicts-lite.pkl',
                        help='Path to output pickle file')
    parser.add_argument('--h5_output',
                        default='data/VG-regions-lite.h5',
                        help='Path to output HDF5 file')

    # OPTIONS
    parser.add_argument('--image_size',
                        default=720, type=int,
                        help='Size of longest edge of preprocessed images')
    parser.add_argument('--max_token_length',
                        default=15, type=int,
                        help="Set to 0 to disable filtering")
    parser.add_argument('--min_token_instances',
                        default=15, type=int,
                        help="When token appears less than this times it will be mapped to <UNK>")
    parser.add_argument('--tokens_type', default='words',
                        help="Words|chars for word or char split in captions")
    parser.add_argument('--max_images', default=-1, type=int,
                        help="Set to a positive number to limit the number of images we process")
    args = parser.parse_args()
    main(args)
