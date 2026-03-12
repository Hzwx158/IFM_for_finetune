CWD="{YOUR WORK DIR}"

DEVICES="0,1"
PORT=12301

MODELS=(
    "base_dino"
    "base_augreg_in1k"
    "base_augreg_in21k"
    "base_sam_in1k"
    "base_orig_in21k"
    # "resnet50.a1_in1k"
)

DATASETS=(
    # "cifar100"
    # "svhn"
    "food101"
)

function merge(){
    echo -e "Merged\nModel: $MERGED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_image.py \
        --model vit_base_patch16 \
        --batch_size 256 --cls_token \
        --finetune "$CWD/pretrained_checkpoint/${MERGED}.pth" \
        --dist_eval --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_merge_before_tune_0.99_blr_0.01/$MERGED  \
        --drop_path 0.0  --blr 0.01 \
        --dataset $dataset \
        --fulltune --merge_before_finetune \
        --mask_dict "$CWD/pretrained_checkpoint/${MASK}.pth"
}
function tune_value_only(){
    echo -e "Tune Value Only\nModel: $CONVERTED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_image.py \
        --model vit_base_patch16 \
        --batch_size 256 --cls_token \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --dist_eval --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_fulltune_value_only_include_v_blr_0.01/${CONVERTED}  \
        --drop_path 0.0  --blr 0.01 \
        --dataset $dataset \
        --fulltune --tune_value_only
}
function merge_and_ffn(){
    echo -e "Merged and FFN\nModel: $MERGED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_image.py \
        --model vit_base_patch16 \
        --batch_size 256 --cls_token \
        --finetune "$CWD/pretrained_checkpoint/${MERGED}.pth" \
        --dist_eval --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_merge_before_tune_0.99_ffn_blr_0.01/$MERGED  \
        --drop_path 0.0  --blr 0.01 \
        --dataset $dataset \
        --fulltune --merge_before_finetune \
        --mask_dict "$CWD/pretrained_checkpoint/${MASK}.pth" \
        --ffn_adapt --ffn_num 1
}
function ffn(){
    echo -e "FFN Adapter\nModel: $CONVERTED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_image.py \
        --model vit_base_patch16 \
        --batch_size 256 --cls_token \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --dist_eval --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_ffn_blr_0.01/${CONVERTED}  \
        --drop_path 0.0  --blr 0.01 \
        --dataset $dataset \
        --ffn_adapt
}
function fulltune(){
    echo -e "Fulltune\nModel: $CONVERTED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_image.py \
        --model vit_base_patch16 \
        --batch_size 256 --cls_token \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --dist_eval --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_fulltune_blr_0.01/${CONVERTED}  \
        --drop_path 0.0  --blr 0.01 \
        --dataset $dataset \
        --fulltune
}
function lora(){
    echo -e "Lora Baseline\nModel: $CONVERTED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_image.py \
        --model vit_base_patch16 \
        --batch_size 256 --cls_token \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --dist_eval --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_lora_blr_0.01/${CONVERTED}  \
        --drop_path 0.0  --blr 0.01 \
        --dataset $dataset \
        --use_lora
}
function merge_cnn(){
    echo -e "Merged\nModel: $MERGED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_cnn.py \
        --model resnet50 \
        --batch_size 256 \
        --finetune "$CWD/pretrained_checkpoint/${MERGED}.pth" \
        --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_merge_before_tune_0.8_blr_0.01/$MERGED  \
        --blr 0.01 \
        --dataset $dataset \
        --merge_before_finetune \
        --mask_dict "$CWD/pretrained_checkpoint/${MASK}.pth"
}
function full_cnn(){
    # echo -e "Merged\nModel: $MERGED\nDataset: $dataset"
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_cnn.py \
        --model resnet50 \
        --batch_size 256 \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_fulltune_blr_0.01/${CONVERTED}  \
        --blr 0.01 \
        --dataset $dataset 
}
function parallel_cnn(){
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_cnn.py \
        --model resnet50 \
        --batch_size 256 \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_parallel_only_new_blr_0.01/${CONVERTED}  \
        --blr 0.01 \
        --dataset $dataset \
        --mask_dict "$CWD/pretrained_checkpoint/${MASK}.pth" \
        --parallel_only_new 
        # --parallel_baseline \
}
function expand_cnn(){
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=$PORT \
        main_cnn.py \
        --model resnet50 \
        --batch_size 256 \
        --finetune "$CWD/pretrained_checkpoint/${CONVERTED}.pth" \
        --data_path $CWD/data \
        --output_dir $CWD/result/${dataset}_expand_blr_only_new_0.01/${CONVERTED} \
        --blr 0.01 \
        --dataset $dataset \
        --mask_dict "$CWD/pretrained_checkpoint/${MASK}.pth" \
        --expand 1
}

for MODEL_PRETRAINED in ${MODELS[@]}
do
    if [[ $MODEL_PRETRAINED =~ "resnet" ]]; then
        CONVERTED="${MODEL_PRETRAINED}_converted"
        MERGED="${CONVERTED}_merge_0.99_no_qk"
        MASK="${CONVERTED}_mask_dict_0.99_no_qk"
    else
        CONVERTED="${MODEL_PRETRAINED}_converted"
        MERGED="${CONVERTED}_merge_0.99_no_qk"
        MASK="${CONVERTED}_mask_dict_0.99_no_qk"
    fi

    for dataset in ${DATASETS[@]} 
    do
        if [[ $1 = "merge" ]]; then
            merge
        elif [[ $1 = "tvo" ]]; then
            tune_value_only
        # elif [[ $1 = "matvo" ]]; then
        #     merge_and_tune_value_only
        elif [[ $1 = "maf" ]]; then
            merge_and_ffn
        elif [[ $1 = "ffn" ]]; then
            ffn
        elif [[ $1 = "lora" ]]; then
            lora
        elif [[ $1 = "all" ]]; then
            merge
            # tune_value_only
            # ffn
            # merge_and_ffn
            # fulltune
            # lora
        elif [[ $1 = "cnn" ]]; then
            merge_cnn
            full_cnn
            expand_cnn
            parallel_cnn
        fi
    done
done