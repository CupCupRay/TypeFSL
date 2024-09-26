import os
import re
import sys
import time
import json
import string

possible_arch = ["arm-32", "arm-64", "mips-32", "mips-64", "x86-32", "x86-64"]
possible_opt = ["O0", "O1", "O2", "O3", "bcfobf", "cffobf", "subobf"]

special_type = ["void"]
prim_type = ["long long", "long double", "long", "double", "float", "char", "short", "int"] # , 
agg_type = ["struct", "union", "enum", "array"] # , 

MINIMUM = 15

def set_default(obj):
    if isinstance(obj, set):
        obj = list(obj)
    if isinstance(obj, dict):
        for key in obj:
            value = obj[key]
            obj[key] = set_default(value)
    return obj


def strip_blank(input):
    if not re.search('[a-zA-Z]+', input):
        return "Unknown"
    
    while(' *' in input):
        input = input.replace(' *', '*')

    ele = input.split(' ')
    while('' in ele):
        ele.remove('')
    return ' '.join(ele)


def identify_common(type_name):
    if type_name == "void*":
        return True
    type_name = type_name.strip('*')
    for token in prim_type:
        if token == type_name or 'unsigned ' + token == type_name:
            return True 
    return False


def clean_data_type(definition, res=''):
    definition = definition.replace('\t', '')
    # Deal with aggregate
    for keyword in agg_type:
        definition = definition.replace(keyword, '').strip()

    # Deal with inline function argument
    if definition.count('(') >= 2 and definition.count(')') >= 2:
        definition = definition[:definition.find(')')].replace('(', '')

    # Check void
    if "void" in definition and '*' in definition[definition.find("void"):]:
        return "void*"
    
    # Check "const" (?) For now, we ignore the "const"
    definition = definition.replace('const', '').strip()
    
    # Check "unsigned"
    UNSIGNED_FLAG = 0
    if "unsigned" in definition:
        definition = definition.replace('unsigned', '').strip()
        UNSIGNED_FLAG = 1

    FLAG = 0
    # Check prim type
    blank_index = len(definition)
    while(blank_index >= 0 and FLAG == 0):
        blank_index = definition.rfind(' ', 0, blank_index)
        if blank_index == -1:
            break
        name = definition[:blank_index].strip('*')
        for t in prim_type:
            # Check normal common type
            if t == name:
                res += t + ' '
                FLAG += 1
                break
    
    if FLAG > 0:
        # Check "pointer"
        # for _ in arg.count('*')
        if '*' in definition:
            res += '*'
        if UNSIGNED_FLAG:
            res = 'unsigned ' + res
        return strip_blank(res)
    # If not matches any prim type, check others
    elif FLAG == 0:
        end = max(definition.find(' '), definition.find('*'))
        if end == -1: 
            processed_arg = definition
        else: 
            processed_arg = definition[:end + 1]
        if UNSIGNED_FLAG:
            processed_arg = 'unsigned ' + processed_arg
        return strip_blank(processed_arg)
    else: print('ERROR: Something wrong with the implementation of clean_data_type')


def generate_label(data_path, gt_path, arch='x86-64', opt='O0', mode='asm'):
    special_functions = dict()
    type_vocabulary = dict()

    # Collect the ground truth files
    gt_files = dict()
    for parent, dirs, files in os.walk(gt_path):
        for file in files:
            project_name = file.replace('.json', '')
            gt_files[project_name] = os.path.join(parent, file)

    temp_file = open(data_path + 'raw_' + mode + '_' + arch + '_' + opt + '.json', mode='w')
    # Collect the processed files
    for parent, dirs, files in os.walk(data_path):
        # Handle different arch, opt-level
        if not parent.endswith(opt) or ('/' + arch + '/') not in parent:
            continue
    
        print("Start to handle dir", parent)
        for f in files:
            # Handle the specific code (asm/mlil/hlil)
            if not f.startswith(mode + '_var_slice'):
                continue
            
            # print('Generating the label (' + mode + ' code) for file:', f)
            target = os.path.join(parent, f)

            # Traverse the ground truth
            target_gt = None
            for project in gt_files:
                if project in parent:
                    target_gt = gt_files[project]
            if target_gt is None:
                for project in gt_files:
                    if project[:project.rfind('-')] in parent:
                        target_gt = gt_files[project]

            if target_gt is None or not target_gt or not os.path.exists(target_gt):
                print('ERROR: Cannot find ground truth file', target_gt)
                continue
            
            # Start to collect the ground truth
            arg_slice = json.load(open(target, mode='r', encoding='utf-8'))
            arg_type = json.load(open(target_gt, mode='r', encoding='utf-8'))
            
            # Handle each function
            for func_name in arg_slice:
                func = arg_slice[func_name]
                if len(func) <= 1: 
                    continue
                
                # Find corresponding ground truth
                gt_func = None
                if func_name in arg_type:
                    gt_func = arg_type[func_name]                        
                if gt_func is None:
                    continue

                # Sort function
                temp_func = list()
                for arg in func:
                    if not re.match(r"arg[\d+]", arg) or not func[arg]:
                        continue
                    for inst in func[arg]:
                        if (' ' + arg + ' ') in inst[1]:
                            temp_func.append([arg, int(inst[0], 16)])
                            break
                    
                temp_func.sort(key=lambda x: x[1])
                FLAG, SKIP_FLAG = False, False
                if len(temp_func) >= 1 and 'arg1' != temp_func[0][0]:
                    FLAG = True
                """
                for i in range(len(temp_func)):
                    if 'arg' + str(i + 1) != temp_func[i][0]:
                        FLAG = True
                        if (target + '=>' + func_name) not in special_functions:
                            special_functions[target + '=>' + func_name] = temp_func
                            """
                
                all_args = list()
                for i in range(len(temp_func)):
                    all_args.append(temp_func[i][0])

                for i in range(len(all_args)):
                    if ('arg' + str(i + 1)) not in all_args:
                        SKIP_FLAG = True
                
                for arg_name in all_args:
                    for inst in func[arg_name]:
                        if 'xmm' in inst[-1] and FLAG:
                            SKIP_FLAG = True
                            if (target + '=>' + func_name) not in special_functions:
                                special_functions[target + '=>' + func_name] = temp_func

                if SKIP_FLAG: 
                    continue
                
                # Generate label
                arg_type_list = list()
                if 'arg_type' in gt_func:
                    arg_type_list = gt_func['arg_type']
                if not arg_type_list:
                    # print('WARNING: Cannot find gruond truth for', arg_name, 'in', func_name)
                    continue
                if len(arg_type_list) != len(all_args):
                    # if func_name not in special_functions:
                        # special_functions[func_name] = [target, target_gt]
                    # print('WARNING: Ground truth does not match for', func_name, 'in', target, 'with', target_gt)
                    continue
                
                for arg_name in all_args:
                    # arg_name = sort_func[my_arg]
                    # if arg_name != my_arg: 
                        # print('WARNING! Failed to match argment:', sort_func)
                    arg_index = int(arg_name.replace('arg', '')) - 1

                    # Get code snippets
                    code_sequence = list()
                    il_sequence = list()

                    for inst in func[arg_name]:
                        if len(inst) > 2:
                            il_sequence.append(inst[1])
                        code_sequence.append(inst[-1])
                        
                    code_sequence = ' '.join(code_sequence).split(' ')
                    # il_sequence = ' '.join(il_sequence).split(' ')
                    code_sequence = [x for x in code_sequence if x != '']
                    il_sequence = [x for x in il_sequence if x != '']

                    if code_sequence == []:
                        continue
                    
                    # Record the type
                    original_type = arg_type_list[arg_index]
                    gt_type = clean_data_type(arg_type_list[arg_index])
                    if gt_type not in type_vocabulary:
                        temp_dict = dict()
                        temp_dict['count'] = 1
                        temp_dict['include'] = [original_type]
                        # Do record
                        type_vocabulary[gt_type] = temp_dict
                    else:
                        temp_dict = dict()
                        ori_count = type_vocabulary[gt_type]['count']
                        included = type_vocabulary[gt_type]['include']

                        temp_dict['count'] = ori_count + 1
                        if original_type not in included:
                            included.append(original_type)
                        temp_dict['include'] = included
                        # Do record 
                        type_vocabulary[gt_type] = temp_dict
                    
                    data = {"text": code_sequence, 
                            "il_text": [arg_name] + il_sequence, 
                            "label": gt_type,
                            "File": target,
                            "Func": func_name}
                    json.dump(data, temp_file)
                    temp_file.write('\n')
    temp_file.close()

    print('Total inconsistent functions for', arch, opt, ':', len(special_functions))
    with open(data_path + 'inconsistence_' + mode + '_' + arch + '_' + opt + '.json', mode='w') as inconsistence_log:
        json.dump(special_functions, inconsistence_log, indent=4)
    
    # Obtain the statics result (sort by first letter)
    type_index, type_count, index = dict(), list(), 0

    for var in type_vocabulary:
        type_count.append((var, type_vocabulary[var]['count']))
    type_list = sorted(type_count, key=lambda x: x[0], reverse=True)
    # Set index according to quantity from most to least
    for var in type_list:
        if identify_common(var[0]) and var[0] not in type_index and var[1] > MINIMUM:
            type_index[var[0]] = index
            index += 1
    # Collect the rare types
    for var in type_list:
        if 'Unknown' in var or var[1] <= MINIMUM or var[0] in type_index:
            continue
        type_index[var[0]] = index
        index += 1

    with open(data_path + 'label_' + mode + '_' + arch + '_' + opt + '.json', mode='w') as st_f:
        json.dump(type_index, st_f, indent=4)
        st_f.write('\n')
        json.dump(type_vocabulary, st_f, indent=4)

    # Write the complete label 
    with open(data_path + mode + '_' + arch + '_' + opt + '.json', mode='w') as result_file:
        for line in open(data_path + 'raw_' + mode + '_' + arch + '_' + opt + '.json', mode='r'):
            line = line.strip('\n')
            try:
                sample = json.loads(line)
                type_name = sample["label"]
                if type_name in type_index:
                    data = {"text": sample["text"], "il_text": sample["il_text"], "label": type_index[type_name]}
                    json.dump(data, result_file)
                    result_file.write('\n')
            except Exception as e:
                continue

            

def get_label(result_root, arch='x86-64', opt='O0', mode='asm', policy='normal'):
    # Mark the label and output json format sample
    print('Processing the label...')
    variable_slices = result_root + 'processed-data.' + policy + '/'
    ground_truth = result_root + 'ground-truth/'
    assert os.path.exists(variable_slices)
    assert os.path.exists(ground_truth)
    
    if 'obf' in opt and arch != 'x86-64':
        return 0
    
    T1 = time.perf_counter()
    generate_label(variable_slices, ground_truth, arch=arch, opt=opt, mode=mode)
    T2 = time.perf_counter()
    total_time = (T2 - T1)
    
    # Time 
    time_effi = {"Total_time(s)": total_time}
    print(time_effi)
    with open(result_root + 'processed-data.' + policy + '/' + 'time.json', 'a') as time_result:
        time_result.write('For all files in mode ' + mode + ' and under root: ' + arch + '/' + opt + '/\n')
        json.dump(time_effi, time_result, indent=4)
    