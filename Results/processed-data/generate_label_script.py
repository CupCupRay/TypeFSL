import os
import json


def write_each_arch_opt(py_output, bash_out, raw_input, label_type):
    # Function def
    raw_title = raw_input[raw_input.rfind('/') + 1:raw_input.rfind('.')]
    title = raw_title.replace('-', '_')
    py_output.write('def get_' + title + '_classes():\n' +
                    '    # @return list of classes associated with each split.\n\n')

    # Contents
    raw_labels = open(raw_input, mode='r').read()
    raw_labels = raw_labels.split('}')[0] + '}'
    label_dict = json.loads(raw_labels)
    py_output.write('    label_dict = ')
    json.dump(label_dict, py_output, indent=8)

    # Dataset split
    py_output.write('\n')
    py_output.write('    train_classes = list(range(0, ' + str(len(label_dict)) + ', 3)) \n')
    py_output.write('    val_classes = list(range(1, ' + str(len(label_dict)) + ', 3)) \n')
    py_output.write('    test_classes = list(range(2, ' + str(len(label_dict)) + ', 3)) \n\n')
    py_output.write('    return train_classes, val_classes, test_classes, label_dict\n\n')

    # Write bash files
    arch = raw_title.split('_')[2]
    opt_lv = raw_title.split('_')[3]

    bash_out.write('elif [[ "$1" == "' + arch + '" ]] && [[ "$2" == "' + opt_lv + '" ]]; then \n')
    bash_out.write('    n_train_class=' + str(len(range(0, len(label_dict), 3))) + '\n')
    bash_out.write('    n_val_class=' + str(len(range(1, len(label_dict), 3))) + '\n')
    bash_out.write('    n_test_class=' + str(len(range(2, len(label_dict), 3))) + '\n')
    bash_out.write('    My_way=' + str(len(range(2, len(label_dict), 3)) - 1) + '\n')

    return 0


if __name__ == '__main__':
    for root, dirs, files in os.walk('./'):
        if 'labels_' not in root:
            continue

        dataset_type = root[root.rfind('/'):]
        print('Start to write', dataset_type)

        scripts = open('./' + dataset_type + '.py', mode='w')
        bash = open('./' + dataset_type + '.sh', mode='w')
        for file in files:
            if 'label_' not in file:
                continue
            write_each_arch_opt(scripts, bash, root + '/' + file, dataset_type)

        print('Complete writing', dataset_type)
        scripts.close()
        bash.close()
