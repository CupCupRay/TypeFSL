#!/bin/bash
# USAGE: bash ./train_asm.sh [arm-32, arm-64, mips-32, mips-64, x86-32, x86-64]
#                            [O0, O1, O2, O3, bcfobf, cffobf, subobf]
#                            [N-way] 
#                            [K-shot]
dataset=asm_$1_$2
type=normal
data_path=./DS/data.${type}/${dataset}.json
wv=./DS/asm.en.vec
result_path=./FSL_results/asm_${type}_$1_$2_$3_way_$4_shot.json
BOOT=1

if [[ "$1" == "arm-32" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=254
    n_val_class=253
    n_test_class=253
    My_way=252
elif [[ "$1" == "arm-32" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=194
    n_val_class=194
    n_test_class=194
    My_way=193
elif [[ "$1" == "arm-32" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=174
    n_val_class=174
    n_test_class=173
    My_way=172
elif [[ "$1" == "arm-32" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=162
    n_val_class=162
    n_test_class=161
    My_way=160
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
    n_train_class=252
    n_val_class=252
    n_test_class=252
    My_way=251
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=202
    n_val_class=202
    n_test_class=201
    My_way=200
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=184
    n_val_class=183
    n_test_class=183
    My_way=182
elif [[ "$1" == "mips-32" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=170
    n_val_class=170
    n_test_class=170
    My_way=169
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=174
    n_val_class=174
    n_test_class=173
    My_way=172
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=118
    n_val_class=117
    n_test_class=117
    My_way=116
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=112
    n_val_class=112
    n_test_class=111
    My_way=110
elif [[ "$1" == "mips-64" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=97
    n_val_class=97
    n_test_class=97
    My_way=96
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=228
    n_val_class=227
    n_test_class=227
    My_way=226
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=191
    n_val_class=191
    n_test_class=190
    My_way=189
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=176
    n_val_class=176
    n_test_class=175
    My_way=174
elif [[ "$1" == "x86-32" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=160
    n_val_class=160
    n_test_class=160
    My_way=159
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "bcfobf" ]]; then 
    n_train_class=214
    n_val_class=214
    n_test_class=214
    My_way=213
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "cffobf" ]]; then 
    n_train_class=204
    n_val_class=204
    n_test_class=204
    My_way=203
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O0" ]]; then 
    n_train_class=260
    n_val_class=260
    n_test_class=260
    My_way=259
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O1" ]]; then 
    n_train_class=208
    n_val_class=208
    n_test_class=207
    My_way=206
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O2" ]]; then 
    n_train_class=188
    n_val_class=188
    n_test_class=187
    My_way=186
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "O3" ]]; then 
    n_train_class=171
    n_val_class=171
    n_test_class=170
    My_way=169
elif [[ "$1" == "x86-64" ]] && [[ "$2" == "subobf" ]]; then 
    n_train_class=215
    n_val_class=215
    n_test_class=215
    My_way=214
else
    BOOT=0
	echo "Please input the expected arch [arm-32, arm-64, mips-32, mips-64, x86-32, x86-64] and opt-level [O0, O1, O2, O3, (x64 Only: bcfobf, cffobf, subobf)]."
fi

if [[ "$3" != "-" ]]; then
    My_way=$3
fi

if [[ $BOOT == 1 ]]; then
    python DS/src/main.py \
        --cuda 2 \
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
