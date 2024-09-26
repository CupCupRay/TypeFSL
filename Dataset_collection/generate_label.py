import numpy as np
from preprocessing import label
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

### USAGE: python generate_dataset.py -ar [x86-64] -ol [O0] -m [asm]

if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--arch', '-ar', required=False, type=str,
                        default='x86-64', help='Which architecture of database is used.')
    parser.add_argument('--result_path', '-rp', required=False, type=str,
                        default='../Result/', help='Where to write result.')
    parser.add_argument('--opt_level', '-ol', required=False, type=str,
                        default='O0', help='Which optimization level to use.')
    parser.add_argument('--mode', '-m', required=False, type=str,
                        default='asm', help='Which kind of input format.')
    parser.add_argument('--policy', '-p', required=False, type=str,
                        default='normal', help='Which kind of input format.')
    
    args = parser.parse_args()
    data_arch = args.arch
    result_root = args.result_path
    opt_level = args.opt_level
    mode = args.mode
    policy = args.policy

    if policy == 'normal' or policy == 'inter':
        # Normal generation, generate [prim(*)] and [user-defined(*)]
        label.get_label(result_root, arch=data_arch, opt=opt_level, mode=mode, policy=policy)