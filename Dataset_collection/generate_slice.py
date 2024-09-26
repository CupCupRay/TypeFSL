import os
import json
import numpy as np
import preprocessing.slice as slice_tech
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

possible_arch = ["arm-32", "arm-64", "mips-32", "mips-64", "x86-32", "x86-64"]
possible_opt = ["O0", "O1", "O2", "O3", "bcfobf", "cffobf", "subobf"]


def get_FileSize(file_path):
    fsize = os.path.getsize(file_path)
    fsize = fsize/float(1024 * 1024)
    return round(fsize, 2)


def program_slicing(data_arch, data_root, result_root, used_package, mode='asm', policy='normal'):
    process_data_path = 'processed-data.' + policy + '/'
    if not data_root.endswith('/'):
        data_root = data_root + '/'
    if not result_root.endswith('/'):
        result_root = result_root + '/'

    # Define the analysis file folder
    if data_arch == 'ALL':
        data_path = data_root
    elif used_package == 'ALL':
        data_path = data_root + data_arch
    else: data_path = data_root + data_arch + used_package
    if not os.path.exists(data_path):
        print("ERROR: Wrong path for the data source", data_path)
        return -1

    # Collect the files
    file_lst = []
    for parent, dirs, files in os.walk(data_path):
        for f in files:
            file_lst.append(os.path.join(parent, f))
    
    file_dict = dict()
    print('Preparation...')
    for f in file_lst:
        root_path = f[:f.rfind('/')]
        second_level_path = root_path[root_path.rfind('/') + 1:]
        root_path = root_path[:root_path.rfind('/')]
        first_level_path = root_path[root_path.rfind('/') + 1:]
        # assert first_level_path in possible_arch

        folder_path = first_level_path + '/' + second_level_path
        if folder_path not in file_dict:
            file_dict[folder_path] = list()
        file_dict[folder_path].append(f)

    print('Vairable slices generation...')
    total_time, total_func_num = 0, 0
    for folder_name in file_dict:
        # Find the arg source folder
        target_folder = result_root + process_data_path + folder_name + '/'
        
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        temp_files = file_dict[folder_name]
        for file in temp_files:
            """
            if 'chmod' not in file:
                continue
            # Check the file size
            if get_FileSize(file) > 10:
                print('FILE:', file, 'IS TOO LARGE, SKIP.')
                continue
            if os.path.exists(target_folder + '_var_slice' + temp_files + '.json'):
                print('Already handled ' + folder_name + ', skip to next.')
                continue
            """
            
            f_time = slice_tech.try_program_slicing(result_root + process_data_path, folder_name, file, mode=mode, policy=policy)
            total_time += f_time["Total_time(s)"]
            total_func_num += f_time["Func_num"]
    # Time 
    time_effi = {"Ave_time(ms)": (total_time / total_func_num) * 1000, "Func_num": total_func_num, "Total_time(s)": total_time}
    print(time_effi)
    with open(result_root + process_data_path + 'time.json', 'a') as time_result:
        time_result.write('For all files in mode ' + mode + ' and under root: ' + data_path + '\n')
        json.dump(time_effi, time_result, indent=4)
    return time_effi


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--data_path', '-dp', required=False, type=str,
                        default='../Dataset/Raw_Dataset/x86-64/coreutils-8.32-O0/', help='Path of the binary file dataset.')
    parser.add_argument('--result_path', '-rp', required=False, type=str,
                        default='../Result/', help='Where to write result.')
    parser.add_argument('--mode', '-m', required=False, type=str,
                        default='asm', help='Which kind of output will be used [asm/mlil/hlil].')
    parser.add_argument('--policy', '-p', required=False, type=str,
                        default='normal', help='Which policy to use for slicing [normal/inter].')
    
    args = parser.parse_args()
    data_path = args.data_path
    result_root = args.result_path
    mode = args.mode
    policy = args.policy
    
    if not data_path.endswith('/'):
        data_path = data_path + '/'
    if not result_root.endswith('/'):
        result_root = result_root + '/'
        
    ele = data_path.split('/')
    if len(ele) >= 4:
        used_package = ele[-2] + '/'
        data_arch = ele[-3] + '/'
        data_root = '/'.join(ele[:-3]) + '/'
        for opt in possible_opt:
            if used_package.endswith(opt + '/'):
                program_slicing(data_arch, data_root, result_root, used_package, mode=mode, policy=policy)
        """
        if data_arch == 'CVE/':
            print("Handle the CVE-related projects:", used_package)
            program_slicing(data_arch, data_root, result_root, used_package, mode=mode, policy=policy)
            """
    else: print("ERROR: Wrong input arguments for slicing")
