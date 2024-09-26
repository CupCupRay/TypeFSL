import torch
import numpy as np
import dataset.labels_normal as labels_normal
import dataset.labels_inter as labels_inter
from dataset.loader import load_agg_dataset, load_normal_dataset, load_inter_dataset


def analyze_df(support, query, args, vocab, error=False):
    support_label = support['label'].tolist()
    query_text = query['text'].tolist()
    query_label = query['label'].tolist()
    _, inv_Q = torch.unique(torch.from_numpy(query['label']), sorted=True, return_inverse=True)
    YQ = torch.arange(start=0, end=args.way)[inv_Q]
    mask = np.ones((len(YQ), args.way), dtype=int)

    # Num to index
    label_ni = dict()
    for num, index in zip(query_label, YQ):
        label_ni[num] = int(index)

    # Obtain the data type 
    label_dict = None
    if args.dataset_type == 'agg':
        _, _, _, label_dict = load_agg_dataset(args)
    elif args.dataset_type == 'normal':
        _, _, _, label_dict = load_normal_dataset(args)
    elif args.dataset_type == 'inter':
        _, _, _, label_dict = load_inter_dataset(args)
    else: return mask
    
    for i in range(len(query_text)):
        # Obtain the specific assembly codes
        asm = [vocab.itos[int(x)] for x in query_text[i]] 
        # print('Assembly codes:', asm)

        # Find the types name
        types = dict()
        for num in support_label:
            for name in label_dict:
                if label_dict[name] == num and name not in types:
                    types[name] = num

        # Get the label type name
        gt_type = None
        for name in label_dict:
            if label_dict[name] == query_label[i]:
                gt_type = name

        impossible_type = list()
        test, count = 1, 0
        for t_name in types:
            if t_name != gt_type:
                impossible_type.append(t_name)
                count += 1
            if count >= test:
                break

        # Mask the impossible type
        for i_type in impossible_type:
            mask[i][label_ni[types[i_type]]] = 0

        # Check the failed cases of filter
        if mask[i][label_ni[types[gt_type]]] == 0:
            error = True

    return mask, error
