#!/bin/bash
# USAGE: bash ./train_asm.sh [arm-32, arm-64, mips-32, mips-64, x86-32, x86-64]
#                            [O0, O1, O2, O3, bcfobf, cffobf, subobf]
#                            [N-way] 
#                            [K-shot]
dataset=asm_$1_$2
type=inter
data_path=./DS/data.${type}/${dataset}.json
wv=./DS/asm.en.vec
result_path=./FSL_results/asm_${type}_$1_$2_$3_way_$4_shot.json
BOOT=1

if [[ "$1" == "arm-32" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=254
    n_val_class=254
    n_test_class=254
    My_way=253
elif [[ "$1" == "arm-32" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=184
    n_val_class=183
    n_test_class=183
    My_way=182
elif [[ "$1" == "arm-32" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=167
    n_val_class=167
    n_test_class=166
    My_way=165
elif [[ "$1" == "arm-32" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=159
    n_val_class=159
    n_test_class=158
    My_way=157
elif [[ "$1" == "arm-64" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=174
    n_val_class=174
    n_test_class=173
    My_way=172
elif [[ "$1" == "arm-64" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=133
    n_val_class=132
    n_test_class=132
    My_way=131
elif [[ "$1" == "arm-64" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=120
    n_val_class=119
    n_test_class=119
    My_way=118
elif [[ "$1" == "arm-64" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=109
    n_val_class=108
    n_test_class=108
    My_way=107
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=253
    n_val_class=253
    n_test_class=252
    My_way=251
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=193
    n_val_class=193
    n_test_class=193
    My_way=192
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=179
    n_val_class=179
    n_test_class=179
    My_way=178
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=168
    n_val_class=167
    n_test_class=167
    My_way=166
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=174
    n_val_class=174
    n_test_class=173
    My_way=172
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=117
    n_val_class=117
    n_test_class=117
    My_way=116
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=112
    n_val_class=111
    n_test_class=111
    My_way=110
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=97
    n_val_class=96
    n_test_class=96
    My_way=95
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=228
    n_val_class=227
    n_test_class=227
    My_way=226
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=189
    n_val_class=189
    n_test_class=188
    My_way=187
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=174
    n_val_class=174
    n_test_class=174
    My_way=173
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=159
    n_val_class=158
    n_test_class=158
    My_way=157
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "bcfobf" ]]; then 
    n_train_class=212
    n_val_class=212
    n_test_class=212
    My_way=211
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "cffobf" ]]; then 
    n_train_class=203
    n_val_class=203
    n_test_class=203
    My_way=202
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=261
    n_val_class=260
    n_test_class=260
    My_way=259
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=195
    n_val_class=195
    n_test_class=194
    My_way=193
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=181
    n_val_class=181
    n_test_class=180
    My_way=179
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=163
    n_val_class=163
    n_test_class=163
    My_way=162
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "subobf" ]]; then 
    n_train_class=215
    n_val_class=214
    n_test_class=214
    My_way=213
else
    BOOT=0
	echo "Please input the expected arch [arm-32, arm-64, mips-32, mips-64, x86-32, x86-64] and opt-level [O0, O1, O2, O3, (x64 Only: bcfobf, cffobf, subobf)]."
fi

if [[ "$3" != "-" ]]; then
    My_way=$3
fi

if [[ $BOOT == 1 ]]; then
    python DS/src/main.py \
        --cuda 1 \
        --way $My_way \
        --shot $4 \
        --query 25 \
        --mode filter \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --dataset_type=$type \
        --data_path=$data_path \
        --word_vector=$wv \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --result_path $result_path \
        --train_episodes 200 \
        --patience 20 \
        --lr 1e-3 \
        --save \
        --meta_iwf \
        --meta_w_target
else
    exit 1;
fi
