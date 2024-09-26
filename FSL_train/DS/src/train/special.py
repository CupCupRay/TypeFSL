import os
import time
import json
import datetime

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler
from dataset.parallel_filter_sampler import Special_ParallelSampler

def base_test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
                                        num_episodes).get_epoch()

    acc = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc.append(test_one(task, model, args))

    acc = np.array(acc)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                ), flush=True)

    return np.mean(acc), np.std(acc)


def filter_test(test_data, model, args, num_episodes, vocab=None, verbose=True, sampled_tasks=None, Sampler=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        Sampler = Special_ParallelSampler(test_data, args, num_episodes)
        sampled_tasks = Sampler.get_epoch()

    acc = []
    std_acc = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    total_error = list()
    for task in sampled_tasks:
        res, error = test_one(task, model, args)
        total_error += error
        std_acc.append(res[0])
        acc.append(res[-1])

    """
    with open('./false.log', mode='a') as flog:
        for ele in total_error:
            out = {'Ground truth': ele[0], 'Pred': ele[-1]}
            json.dump(out, flog)
            flog.write('\n')
            """

    std_acc = np.array(std_acc)
    acc = np.array(acc)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(std_acc),
                colored("std", "blue"),
                np.std(std_acc),
                ), flush=True)
        
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("[Struct] acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                ), flush=True)

    return [np.mean(std_acc), np.mean(acc)], [np.std(std_acc), np.std(acc)]


def test(test_data, model, args, num_episodes, vocab=None, verbose=True, sampled_tasks=None):
    test_acc, test_std = base_test(test_data, model, args, num_episodes, verbose, sampled_tasks)
    # new_test_acc, new_test_std = filter_test(test_data, model, args, num_episodes, vocab, verbose, sampled_tasks)

    return test_acc, test_std
    # return new_test_acc, new_test_std

def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    mask = None
    if len(task) == 3:
        support, query, mask = task
    else: support, query = task

    # Embedding the document
    XS = model['ebd'](support)
    YS = support['label']

    XQ = model['ebd'](query)
    YQ = query['label']

    # Apply the classifier
    if mask is not None:
        acc, _, error = model['clf'](XS, YS, XQ, YQ, mask)
    else:
        acc, _ = model['clf'](XS, YS, XQ, YQ)

    if mask is not None:
        return acc, error

    return acc
    