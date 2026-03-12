
MODELS=(
    # "dino"
    # "augreg_in1k"
    # "augreg_in21k"
    # "sam_in1k"
    # "orig_in21k"
    "resnet50.a1_in1k"
)

if [[ $1 = "download" ]]; then
    for i in ${MODELS[@]} 
    do
        if [[ $i =~ "resnet" ]]; then
            hf download "timm/$i" --local-dir "./raw_weights/$i"
        else
            hf download \
                "timm/vit_base_patch16_224.$i" \
                --local-dir "./raw_weights/base_$i"
        fi
    done
elif [[ $1 = "train" ]]; then
    for i in ${MODELS[@]} 
    do
        if [[ $i =~ "resnet" ]]; then
            export MODEL_PRETRAINED="$i"
        else
            export MODEL_PRETRAINED="base_$i"
        fi
        python convert.py
        python merge_and_test.py
    done
elif [[ $1 = "dataset" ]]; then
    python prepare_dataset.py
else
    echo "Wrong Argument: $1"
fi