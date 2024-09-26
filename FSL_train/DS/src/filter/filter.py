import re
import time
import json
import torch
import numpy as np
import dataset.labels_normal as labels_normal
import dataset.labels_inter as labels_inter
from dataset.loader import load_normal_dataset, load_inter_dataset

special_type = ["void"]
sign_type = ["unsigned", '']
int_type = ["short", "int", "long", "char"]
prim_type = ["long long", "long double", "long", "double", "float", "char", "short", "int"] # , 
agg_type = ["struct", "union", "enum", "array"] # , 


def data_transfer(target, tokens, mode='r'):
    if mode == 'w' and ('[' not in target or ']' not in target):
        return target.replace(' ', '')
    
    for wt in tokens:
        target = target.replace(' ' + wt + ' ', tokens[wt][-1])
    return target.replace(' ', '')


def get_atom(constraints):
    tokens = dict()
    operations = list()
    arg_name = constraints[0]
    if len(constraints) > 100:
        constraints = constraints[:100]
    for c in constraints[1:]:
        # If-else statement
        if ' if ' in c and ' then ' in c:
            operations.append(c[c.find(' if ') + 3: c.find(' then ')].replace(' ', ''))
        # Assignment
        elif ' = ' in c:
            write_token = c[:c.find(' = ')]
            read_token = c[c.find(' = ') + 3:]

            write_token = data_transfer(write_token, tokens, mode='w')
            read_token = data_transfer(read_token, tokens, mode='r')

            if write_token in tokens:
                temp = tokens[write_token] 
                if read_token not in temp:
                    temp.append(read_token)
                tokens[write_token] = temp
                del temp
            else: tokens[write_token] = [read_token]
    
    return arg_name, tokens, operations


def check_struct(arg_name, tokens, operations):
    temp_types = list()
    FLAG = False
    for wt in tokens:
        if '[' in wt and arg_name in wt and ('+' in wt or '-' in wt) and '].' in wt:
            if wt.find('[') < wt.find(arg_name) < wt.find('].'):
                if '+' in wt[wt.find(arg_name): wt.find('].')] or \
                '-' in wt[wt.find(arg_name): wt.find('].')]:
                    FLAG = True
                    break
        res = [True if (('[' in rt and arg_name in rt and ('+' in rt or '-' in rt)and '].' in rt) and 
               (rt.find('[') < rt.find(arg_name) < rt.find('].')) and
               ('+' in rt[rt.find(arg_name): rt.find('].')] or '-' in rt[rt.find(arg_name): rt.find('].')]))
               else False for rt in tokens[wt]]
        if True in res:
            FLAG = True
            break
    
    for ope in operations:
        if '[' in ope and arg_name in ope and ('+' in ope or '-' in ope) and '].' in ope:
            if ope.find('[') < ope.find(arg_name) < ope.find('].'):
                if '+' in ope[ope.find(arg_name): ope.find('].')] or \
                '-' in ope[ope.find(arg_name): ope.find('].')]:
                    FLAG = True
                    break

    if FLAG: temp_types = ([x for x in prim_type] + 
                           [x + '*' for x in prim_type] + 
                           ['unsigned ' + x for x in prim_type] + 
                           ['unsigned ' + x + '*' for x in prim_type])

    return temp_types


def check_ptr(arg_name, ins):
    if '[' in ins and arg_name in ins and '].' in ins:
        if ins.find('[') < ins.find(arg_name) < ins.find('].'):
            return True
    return False


def check_include_ptr(arg_name, tokens, operations):
    temp_types = list()
    FLAG = False
    for wt in tokens:
        if check_ptr(arg_name, wt):
            FLAG = True
            break
        res = [check_ptr(arg_name, rt) for rt in tokens[wt]]
        if True in res:
            FLAG = True
            break
    
    for ope in operations:
        if check_ptr(arg_name, ope):
            FLAG = True
            break

    if FLAG: temp_types = [x for x in prim_type] + ['unsigned ' + x for x in prim_type]

    return temp_types


def check_decimal(arg_name, tokens, operations):
    temp_types = list()
    for wt in tokens:
        for rt in tokens[wt]:
            if arg_name in wt and rt.startswith('-'):
                temp_types += ['unsigned ' + x for x in int_type] + ['unsigned ' + x + '*' for x in int_type]
            nums = re.findall(r"[-+]?\d+\.?\d*[eE]?[-+]?\d*", rt)
            if nums != [] and (arg_name in wt or arg_name in rt):
                for num in nums:
                    if re.match(r"\d+\.\d+", num):
                        temp_types += ([x for x in int_type] + [x + '*' for x in int_type] + ['unsigned ' + x for x in int_type] + ['unsigned ' + x + '*' for x in int_type])
                    if '.' in num and len(num[num.find('.') + 1:]) > 8:
                        temp_types += ['float', 'float*']

    return temp_types


def check_shift(arg_name, tokens, operations):
    temp_types = list()
    for wt in tokens:
        for rt in tokens[wt]:
            if not check_ptr(arg_name, rt) and ('<<' in rt or '>>' in rt):
                temp_types += ([x + '*' for x in (prim_type + agg_type)] +
                               ['unsigned ' + x + '*' for x in (prim_type + agg_type)] +
                               ['long double', 'double', 'float'])
            elif check_ptr(arg_name, rt) and ('<<' in rt or '>>' in rt):
                temp_types += ['long double', 'double', 'float']
    
    return temp_types


def find_impossible(std_types, asm, il):
    temp_types = list()
    # print("Asm code:", asm)
    # print("MLIL codes:", il)
    """
    with open('./il.test', mode='a') as il_test:
        json.dump({'il': il, 'gt': gt_type}, il_test)
        il_test.write('\n')
        """
    # value_set = get_atom([x.replace(' ', '') for x in il])
    arg, ts, opers = get_atom(il)
    """
    # print("Length:", len(tokens))
    with open('./token.log', mode='a') as token_log:
        json.dump({'MLIL': il, 'tokens': ts, 'Operations': opers}, token_log, indent=4)
        token_log.write('\n')
        """
    
    # TODO: Other rules
    temp_types += check_include_ptr(arg, ts, opers)
    # temp_types += check_struct(arg, ts, opers)
    temp_types += check_decimal(arg, ts, opers)
    # temp_types += check_shift(arg, ts, opers)
    
    temp_types = list(set(temp_types))
    # print("Ans:", temp_types)
    """
    if gt_type.replace('train_', '').replace('val_', '').replace('test_', '') in temp_types:
        print('WARNING!', il)
        print('Ans:', temp_types)
        print('Ground truth', gt_type)
        """

    # Final result
    im_types = list()
    for temp in temp_types:
        for t_name in std_types:
            if t_name == temp or \
            t_name == 'train_' + temp or \
            t_name == 'val_' + temp or \
            t_name == 'test_' + temp:
                im_types.append(t_name)
    """
    if im_types != []:
        print('GT:', gt_type, 'Filter:', im_types)
        """
    return im_types


def analyze_df(query, args):
    _, inv_Q = torch.unique(torch.from_numpy(query['label']), sorted=True, return_inverse=True)
    YQ = torch.arange(start=0, end=args.way)[inv_Q]
    # _, inv_Q = np.unique(query['label'], return_inverse=True)
    # YQ = np.arange(start=0, stop=args.way)[inv_Q]
    mask = np.ones((len(YQ), args.way), dtype=int)

    # Num to index
    label_ni = dict()
    for num, index in zip(query['label'], YQ):
        label_ni[num] = int(index)

    # Obtain the data type 
    label_dict = None
    if args.dataset_type == 'normal':
        _, _, _, label_dict = load_normal_dataset(args)
    elif args.dataset_type == 'inter':
        _, _, _, label_dict = load_inter_dataset(args)
    else: return mask
    
    # Find the types name
    types = dict()
    for num in query['label']:
        for name in label_dict:
            if label_dict[name] == num and name not in types:
                types[name] = num

    for i in range(len(query['text'])):
        # Obtain the specific assembly codes
        # asm = [vocab.itos[int(x)] for x in query['text'][i]] 
        asm = query['raw'][i]
        il = query['il_text'][i]
        # print('Assembly codes:', asm)

        # Get the label type name
        gt_type = None
        for name in label_dict:
            if label_dict[name] == query['label'][i]:
                gt_type = name

        # TODO: Use rule to find impossible types
        impossible_type = find_impossible(types, asm, il) # , gt_type
        """
        test, count = 10, 0
        for t_name in types:
            if t_name != gt_type:
                impossible_type.append(t_name)
                count += 1
            if count >= test:
                break
                """

        # Mask the impossible type
        for i_type in impossible_type:
            mask[i][label_ni[types[i_type]]] = 0

        # Check the failed cases of filter
        if mask[i][label_ni[types[gt_type]]] == 0:
            print(gt_type, impossible_type)
            print(il)
        
    return mask


def analyze_df(query, args):
    # Obtain the data type 
    label_dict = None
    if args.dataset_type == 'normal':
        _, _, _, label_dict = load_normal_dataset(args)
    elif args.dataset_type == 'inter':
        _, _, _, label_dict = load_inter_dataset(args)
    else: return False
    
    # Find the types name
    types = dict()
    for num in query['label']:
        for name in label_dict:
            if label_dict[name] == num and name not in types:
                types[name] = num

    FLAG = False
    
    for i in range(len(query['text'])):
        # Get the label type name
        gt_type = None
        for name in label_dict:
            if label_dict[name] == query['label'][i]:
                gt_type = name
        if gt_type is not None:
            for t in ([prim_type] + [x + '*' for x in prim_type] +  ['unsigned ' + x for x in prim_type] +  ['unsigned ' + x + '*' for x in prim_type]):
                if gt_type == t:
                    FLAG = True

    return FLAG
        