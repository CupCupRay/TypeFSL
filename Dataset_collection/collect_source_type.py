import os
import sys
import json
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

### Please first run ./use_ctags.sh
### USAGE: python collect_source_type.py 

def find_arg_type(func_def):
    arg_type = list()

    # Extract arg type
    arg_text = func_def[func_def.find('(') + 1:]
    
    # Collect each arg
    SKIP_FLAG, READ_FLAG, current_arg = 0, 1, ''
    prev = None
    for c in arg_text:
        if prev is not None:
            if prev + c == '/*':
                READ_FLAG -= 1
                current_arg = current_arg[:-1]
            elif prev + c == '*/':
                READ_FLAG += 1
                prev = c
                continue
        if c == '(':
            SKIP_FLAG += 1
        elif c == ')':
            SKIP_FLAG -= 1
        
        # Reach the end of the argument definition
        if SKIP_FLAG == -1:
            break

        if SKIP_FLAG == 0 and c == ',':
            current_arg = current_arg.strip()
            if current_arg: 
                arg_type.append(current_arg)
            current_arg = ''
        elif READ_FLAG > 0: 
            current_arg += c
        prev = c

    current_arg = current_arg.strip()
    if current_arg: 
        arg_type.append(current_arg)

    """
    arg_list = arg_text.split(',')
    for each_arg in arg_list:
        end = max(each_arg.rfind(' '), each_arg.rfind('*'))
        processed_arg = each_arg[:end + 1]
        if end == -1: processed_arg = each_arg
        if processed_arg:
            arg_type.append(processed_arg.strip())
        if each_arg:
            arg_type.append(each_arg.strip())
            """
    return arg_type


def trace_source_def(func_name, line_index, file_source):
    func_def, IN_FUNC = '', False
    line_count, left_b, right_b = 1, 0, 0
    if not os.path.exists(file_source):
        print("ERROR: Cannot find file =>", file_source)
        return func_def
    
    # Try to read file
    try:
        for line in open(file_source, mode='r'):
            line = line.strip('\n').strip()
            # Whether this line is about function definition
            if line_count == line_index: # and func_name in line:
                func_def = line[line.find(func_name):]
                for c in func_def:
                    if c == '(': 
                        left_b += 1
                    elif c == ')': 
                        right_b += 1
                IN_FUNC = True
            elif IN_FUNC: 
                for c in line:
                    # Break the loop
                    if left_b == right_b and left_b > 0: 
                        IN_FUNC = False
                        break
                    if c == '(': 
                        left_b += 1
                    elif c == ')': 
                        right_b += 1
                    func_def += c
            line_count += 1
    except Exception as e:
        print('WARNING: Cannot handle file ' + file_source + ' with utf-8 encoding')
        print(e)
        return ''

    return func_def


def ctags_analysis(path, dst, filename):
    info_dict = dict()
    for line in open(path, mode='r'):
        line = line.strip('\n').strip()
        ele = line.split(' ')
        while('' in ele):
            ele.remove('')

        # Collect each element
        func_name = ele[0]
        identifier = ele[1]
        line_index = ele[2]
        file_source = ele[3]
        func_def = ' '.join(ele[4:])

        if identifier != 'function':
            continue
        if not line_index.isdigit():
            continue

        # When definition is not complete
        if func_def.count('(') != func_def.count(')') or func_def.count('(') == 0:
            func_def = trace_source_def(func_name, int(line_index), file_source)
        # Check again
        if func_def.count('(') != func_def.count(')') or func_def.count('(') == 0:
            print("WARNING: Function definition has error in =>", func_name)
            continue
        
        # Collect the argument type from definition
        temp_dict = dict()
        temp_dict['source'] = file_source[file_source.find(filename.replace('.tags', '')):]
        temp_dict['line'] = line_index
        arg_type = find_arg_type(func_def)
        if arg_type and arg_type is not None: 
            temp_dict['arg_type'] = arg_type

        # Error checking
        record_func = func_name
        if func_name in info_dict:
            """
            count = 1
            record_func = func_name + '_' + str(count)
            while(record_func in info_dict):
                count += 1
                record_func = func_name + '_' + str(count)
                """
            
            if 'arg_type' in info_dict[func_name] and 'arg_type' in temp_dict:
                # Use the version with more details
                prev_dict = info_dict[func_name]
                prev_info, this_info = 0, 0
                for each_arg in prev_dict['arg_type']:
                    prev_info += len(each_arg)
                for each_arg in temp_dict['arg_type']:
                    this_info += len(each_arg)
                if prev_info >= this_info:
                    temp_dict = prev_dict
        info_dict[record_func] = temp_dict
    
    # Write result
    with open(dst + filename.replace('.tags', '.json'), 'w') as result_file:
        # json.dump(function_var_type, result_file)
        json.dump(info_dict, result_file, indent=4)


def batch_analyze(src, dst):
    print("Start to collect the ground truth from ctags")
    for root, dirs, files in os.walk(src):
        for file in files:
            """
            if 'coreutils-8.32' not in file:
                continue
                """
            if file.endswith('.tags') or file.endswith('.cpp') or file.endswith('.h'):
                print("Now handling the file", file)
                ctags_analysis(os.path.join(root, file), dst, file)


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--data_path', '-dp', required=False, type=str,
                        default='../Dataset/Source_Dataset/Index/', help='Path of the binary file dataset.')
    parser.add_argument('--result_path', '-rp', required=False, type=str,
                        default='../Result/ground-truth/', help='Where to write result.')

    args = parser.parse_args()
    data_root = args.data_path
    result_root = args.result_path

    batch_analyze(data_root, result_root)
