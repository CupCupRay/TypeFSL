import os
import sys
import itertools
import collections
import json
from collections import defaultdict

import numpy as np
import torch

import dataset.labels_normal as labels_normal
import dataset.labels_inter as labels_inter

from torchtext.vocab import Vocab, Vectors

from embedding.avg import AVG
from embedding.cxtebd import CXTEBD
from embedding.wordebd import WORDEBD
import dataset.stats as stats
from dataset.utils import tprint

from transformers import BertTokenizer


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500],  # truncate the text to 500 tokens
                'il_text': row['il_text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data


def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)
    il_text = np.array([e['il_text'] for e in data], dtype=object)

    if args.bert:
        tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased', do_lower_case=True)

        # convert to wpe
        vocab_size = 0  # record the maximum token id for computing idf
        for e in data:
            e['bert_id'] = tokenizer.convert_tokens_to_ids(['CLS'] + e['text'] + ['SEP'])
            vocab_size = max(max(e['bert_id'])+1, vocab_size)

        text_len = np.array([len(e['bert_id']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']

            # filter out document with only special tokens
            # unk (100), cls (101), sep (102), pad (0)
            if np.max(text[i]) < 103:
                del_idx.append(i)

        text_len = text_len

    else:
        # compute the max text length
        text_len = np.array([len(e['text']) for e in data])
        max_text_len = max(text_len)

        # initialize the big numpy array by <pad>
        text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                         dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['text'])] = [
                    vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                    for x in data[i]['text']]

            # filter out document with only unk and pad
            if np.max(text[i]) < 2:
                del_idx.append(i)

        vocab_size = vocab.vectors.size()[0]

    text_len, text, il_text, doc_label, raw = _del_by_idx(
            [text_len, text, il_text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'il_text': il_text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
    }

    if 'pos' in args.auxiliary:
        # use positional information in fewrel
        head = np.vstack([e['head'] for e in data])
        tail = np.vstack([e['tail'] for e in data])

        new_data['head'], new_data['tail'] = _del_by_idx(
            [head, tail], del_idx, 0)

    return new_data


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val

def load_normal_dataset(args):
    if args.dataset == 'asm_x86-64_O0':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_O0_classes()
    elif args.dataset == 'asm_x86-64_O1':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_O1_classes()
    elif args.dataset == 'asm_x86-64_O2':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_O2_classes()
    elif args.dataset == 'asm_x86-64_O3':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_O3_classes()
    elif args.dataset == 'asm_x86-32_O0':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_32_O0_classes()
    elif args.dataset == 'asm_x86-32_O1':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_32_O1_classes()
    elif args.dataset == 'asm_x86-32_O2':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_32_O2_classes()
    elif args.dataset == 'asm_x86-32_O3':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_32_O3_classes()
    elif args.dataset == 'asm_arm-32_O0':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_32_O0_classes()
    elif args.dataset == 'asm_arm-32_O1':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_32_O1_classes()
    elif args.dataset == 'asm_arm-32_O2':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_32_O2_classes()
    elif args.dataset == 'asm_arm-32_O3':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_32_O3_classes()
    elif args.dataset == 'asm_arm-64_O0':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_64_O0_classes()
    elif args.dataset == 'asm_arm-64_O1':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_64_O1_classes()
    elif args.dataset == 'asm_arm-64_O2':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_64_O2_classes()
    elif args.dataset == 'asm_arm-64_O3':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_arm_64_O3_classes()
    elif args.dataset == 'asm_mips-32_O0':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_32_O0_classes()
    elif args.dataset == 'asm_mips-32_O1':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_32_O1_classes()
    elif args.dataset == 'asm_mips-32_O2':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_32_O2_classes()
    elif args.dataset == 'asm_mips-32_O3':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_32_O3_classes()
    elif args.dataset == 'asm_mips-64_O0':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_64_O0_classes()
    elif args.dataset == 'asm_mips-64_O1':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_64_O1_classes()
    elif args.dataset == 'asm_mips-64_O2':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_64_O2_classes()
    elif args.dataset == 'asm_mips-64_O3':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_mips_64_O3_classes()
    elif args.dataset == 'asm_x86-64_bcfobf':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_bcfobf_classes()
    elif args.dataset == 'asm_x86-64_cffobf':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_cffobf_classes()
    elif args.dataset == 'asm_x86-64_subobf':
        train_classes, val_classes, test_classes, label_dict = labels_normal.get_label_asm_x86_64_subobf_classes()
    else:
        print('ERROR Dataset:', args.dataset)
        raise ValueError(
            'args.dataset should be one of '
            '[arm-32, arm-64, mips-32, mips-64, x86-32, x86-64] + '
            '[O0, O1, O2, O3, bcfobf (For x64), cffobf (For x64), subobf (For x64)]')
    return train_classes, val_classes, test_classes, label_dict


def load_inter_dataset(args):
    if args.dataset == 'asm_x86-64_O0':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_O0_classes()
    elif args.dataset == 'asm_x86-64_O1':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_O1_classes()
    elif args.dataset == 'asm_x86-64_O2':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_O2_classes()
    elif args.dataset == 'asm_x86-64_O3':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_O3_classes()
    elif args.dataset == 'asm_x86-32_O0':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_32_O0_classes()
    elif args.dataset == 'asm_x86-32_O1':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_32_O1_classes()
    elif args.dataset == 'asm_x86-32_O2':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_32_O2_classes()
    elif args.dataset == 'asm_x86-32_O3':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_32_O3_classes()
    elif args.dataset == 'asm_arm-32_O0':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_32_O0_classes()
    elif args.dataset == 'asm_arm-32_O1':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_32_O1_classes()
    elif args.dataset == 'asm_arm-32_O2':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_32_O2_classes()
    elif args.dataset == 'asm_arm-32_O3':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_32_O3_classes()
    elif args.dataset == 'asm_arm-64_O0':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_64_O0_classes()
    elif args.dataset == 'asm_arm-64_O1':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_64_O1_classes()
    elif args.dataset == 'asm_arm-64_O2':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_64_O2_classes()
    elif args.dataset == 'asm_arm-64_O3':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_arm_64_O3_classes()
    elif args.dataset == 'asm_mips-32_O0':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_32_O0_classes()
    elif args.dataset == 'asm_mips-32_O1':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_32_O1_classes()
    elif args.dataset == 'asm_mips-32_O2':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_32_O2_classes()
    elif args.dataset == 'asm_mips-32_O3':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_32_O3_classes()
    elif args.dataset == 'asm_mips-64_O0':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_64_O0_classes()
    elif args.dataset == 'asm_mips-64_O1':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_64_O1_classes()
    elif args.dataset == 'asm_mips-64_O2':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_64_O2_classes()
    elif args.dataset == 'asm_mips-64_O3':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_mips_64_O3_classes()
    elif args.dataset == 'asm_x86-64_bcfobf':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_bcfobf_classes()
    elif args.dataset == 'asm_x86-64_cffobf':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_cffobf_classes()
    elif args.dataset == 'asm_x86-64_subobf':
        train_classes, val_classes, test_classes, label_dict = labels_inter.get_label_asm_x86_64_subobf_classes()
    else:
        print('ERROR Dataset:', args.dataset)
        raise ValueError(
            'args.dataset should be one of '
            '[arm-32, arm-64, mips-32, mips-64, x86-32, x86-64] + '
            '[O0, O1, O2, O3, bcfobf (For x64), cffobf (For x64), subobf (For x64)]')
    return train_classes, val_classes, test_classes, label_dict


def load_dataset(args):
    if args.dataset_type == 'agg':
        train_classes, val_classes, test_classes, _ = load_agg_dataset(args)
    elif args.dataset_type == 'normal':
        train_classes, val_classes, test_classes, _ = load_normal_dataset(args)
    elif args.dataset_type == 'inter':
        train_classes, val_classes, test_classes, _ = load_inter_dataset(args)

    print("Train class:", len(train_classes), args.n_train_class)
    print("Val class:", len(val_classes), args.n_val_class)
    print("Test class:", len(test_classes), args.n_test_class)
    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    if args.mode == 'finetune':
        # in finetune, we combine train and val for training the base classifier
        train_classes = train_classes + val_classes
        args.n_train_class = args.n_train_class + args.n_val_class
        args.n_val_class = args.n_train_class

    tprint('Loading data')
    all_data = _load_json(args.data_path)

    tprint('Loading word vectors')
    path = os.path.join(args.wv_path, args.word_vector)
    if not os.path.exists(path):
        print("ERROR: Please prepare the assembly language word vectors, not", path)
        sys.exit("Word vector not prepared!")
        """
        # Download the word vector and save it locally:
        tprint('Downloading word vectors')
        import urllib.request
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
            path)"""

    vectors = Vectors(args.word_vector, cache=args.wv_path)
    vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are initialized to zeros): {}').format( num_oov))

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
            all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))


    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, vocab, args)
    tprint('Complete training set.')
    val_data = _data_to_nparray(val_data, vocab, args)
    tprint('Complete validation set.')
    test_data = _data_to_nparray(test_data, vocab, args)
    tprint('Complete testing set.')

    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    stats.precompute_stats(train_data, val_data, test_data, args)

    if args.meta_w_target:
        # augment meta model by the support features
        if args.bert:
            ebd = CXTEBD(args.pretrained_bert,
                         cache_dir=args.bert_cache_dir,
                         finetune_ebd=False,
                         return_seq=True)
        else:
            ebd = WORDEBD(vocab, finetune_ebd=False)

        train_data['avg_ebd'] = AVG(ebd, args)
        if args.cuda != -1:
            train_data['avg_ebd'] = train_data['avg_ebd'].cuda(args.cuda)

        val_data['avg_ebd'] = train_data['avg_ebd']
        test_data['avg_ebd'] = train_data['avg_ebd']

    # if finetune, train_classes = val_classes and we sample train and val data
    # from train_data
    if args.mode == 'finetune':
        train_data, val_data = _split_dataset(train_data, args.finetune_split)

    return train_data, val_data, test_data, vocab
