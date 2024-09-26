dataset=reuters
data_path="data/reuters.json"
n_train_class=15
n_val_class=5
n_test_class=11
if [ "$dataset" = "fewrel" ]; then
    python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode finetune \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --auxiliary pos \
        --meta_iwf \
        --meta_w_target
else
    python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --meta_iwf \
        --meta_w_target
fi

