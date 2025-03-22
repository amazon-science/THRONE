# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

###################################################################################################
# ENV SETUP
###################################################################################################

MODEL="${MODEL:-LLaVAMistral}"
DATASET="${DATASET:-o365_subset}"
ROOT="${ROOT:-$PWD/../}"
PRETRAINED_MODELS="${PRETRAINED_MODELS:-$ROOT/../models}"
export NCCL_IGNORE_DISABLED_P2P=1


###################################################################################################
# MODEL SELECTION
###################################################################################################

set -e
if [ "$MODEL" == "LLaVAMistral" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 8 LLaVA --model_path $PRETRAINED_MODELS/llava-v1.6-mistral-7b --conv_template_name mistral_instruct"
    pythonpath="$ROOT/third_party/LLaVA/" # Set according to the path of the LLaVA repo
elif [ "$MODEL" == "LLaVAv1.3" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 16 LLaVA --model_path $PRETRAINED_MODELS/llava-lora-vicuna-7b-v1.3 --model_base $PRETRAINED_MODELS/vicuna-7b-v1.3 --conv_template_name llava_v1"
    pythonpath="$ROOT/third_party/LLaVA/" # Set according to the path of the LLaVA repo
elif [ "$MODEL" == "LLaVAv1.5" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 16 LLaVA --model_path $PRETRAINED_MODELS/llava-v1.5-7b --conv_template_name llava_v1"
    pythonpath="$ROOT/third_party/LLaVA/" # Set according to the path of the LLaVA repo
elif [ "$MODEL" == "InstructBLIP" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 16 InstructBLIP --model_path $PRETRAINED_MODELS/instructblip-vicuna-7b"
elif [ "$MODEL" == "mPLUGOwl" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 16 mPLUGOwl --model_path $PRETRAINED_MODELS/mplug-owl-llama-7b"
    pythonpath="$ROOT/third_party/mPLUG-Owl/mPLUG-Owl/" # Set according to the path of the mPLUG-Owl repo
elif [ "$MODEL" == "LRVInstruct" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 16 LRVInstruct --model_base $PRETRAINED_MODELS/mplug-owl-llama-7b --model_path $PRETRAINED_MODELS/LRV-Instruction-V2/pytorch_model.bin"
    pythonpath="$ROOT/third_party/mPLUG-Owl/mPLUG-Owl/" # Set according to the path of the mPLUG-Owl repo
elif [ "$MODEL" == "AdapterV2" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 1 LLaMAAdapter --model_name LORA-BIAS-7B --llama_model_path $PRETRAINED_MODELS/LLaMA"
    pythonpath="$ROOT/third_party/LLaMA-Adapter/llama_adapter_v2_multimodal7b/"
elif [ "$MODEL" == "AdapterV2.1" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 1 LLaMAAdapter --model_name LORA-BIAS-7B-v21 --llama_model_path $PRETRAINED_MODELS/LLaMA"
    pythonpath="$ROOT/third_party/LLaMA-Adapter/llama_adapter_v2_multimodal7b/"
elif [ "$MODEL" == "Otter" ]; then
    MODEL_GEN_FLAGS="--per_device_batch_size 8 Otter --model_path $PRETRAINED_MODELS/OTTER-Image-MPT7B"
    pythonpath="$ROOT/third_party/Otter/src/"
elif [ "$MODEL" == "MiniGPT" ]; then
    sed -i 's+"please set this value to .*"+"'$PRETRAINED_MODELS'/vicuna-7b/"+g' $ROOT/MiniGPT-4/minigpt4/configs/models/minigpt4_vicuna0.yaml
    sed -i 's/low_resource: True/low_resource: False/g' $ROOT/MiniGPT-4/eval_configs/minigpt4_eval.yaml
    sed -i 's+please set this value to .* checkpoint+'$PRETRAINED_MODELS'/mini-gpt4/prerained_minigpt4_7b.pth+g' $ROOT/MiniGPT-4/eval_configs/minigpt4_eval.yaml
    MODEL_GEN_FLAGS="--per_device_batch_size 16 MiniGPT --cfg_path $ROOT/MiniGPT-4/eval_configs/minigpt4_eval.yaml --model_path ''"
    pythonpath="$ROOT/third_party/MiniGPT-4/"
elif [ "$MODEL" == "MiniGPTv2" ]; then
    sed -i 's+"please set this value to .*"+"'$PRETRAINED_MODELS'/Llama-2-7b-chat-hf/"+g' $ROOT/MiniGPT-4/minigpt4/configs/models/minigpt_v2.yaml
    sed -i 's/low_resource: True/low_resource: False/g' $ROOT/MiniGPT-4/eval_configs/minigptv2_eval.yaml
    sed -i 's+please set this value to .* checkpoint+'$PRETRAINED_MODELS'/mini-gptv2/checkpoint_stage3.pth+g' $ROOT/MiniGPT-4/eval_configs/minigptv2_eval.yaml
    MODEL_GEN_FLAGS="--per_device_batch_size 1 MiniGPTv2 --cfg_path $ROOT/MiniGPT-4/eval_configs/minigptv2_eval.yaml --model_path ''"
    pythonpath="$ROOT/third_party/MiniGPT-4/"
else
    echo MODEL not recognized: $MODEL >&2
    exit 1
fi



###################################################################################################
# DATA SELECTION
###################################################################################################
if [ "$DATASET" == "coco_full" ]; then
    COCO_FILE="$ROOT/data/coco/annotations/instances_val2017.json"
    COCO_IMAGE="$ROOT/data/coco/val2017/"
    COCO_SUB_FLAGS=""
elif [ "$DATASET" == "o365_subset" ]; then
    COCO_FILE="$ROOT/data/objects365/zhiyuan_objv2_val_fixname.json"
    COCO_IMAGE="$ROOT/data/objects365/"
    COCO_SUB_FLAGS="--coco_subset objects365_subset.txt"
elif [ "$DATASET" == "coco_subset" ]; then # for debugging only; not used in paper
    COCO_FILE="$ROOT/data/coco/annotations/instances_val2017.json"
    COCO_IMAGE="$ROOT/data/coco/val2017/"
    COCO_SUB_FLAGS="--coco_subset coco_subset.txt"
elif [ "$DATASET" == "o365_subset_subset" ]; then # for debugging only; not used in paper
    COCO_FILE="$ROOT/data/objects365/zhiyuan_objv2_val_fixname.json"
    COCO_IMAGE="$ROOT/data/objects365/"
    COCO_SUB_FLAGS="--coco_subset objects365_subset_subset.txt"
else
    echo DATASET not recognized: $DATASET >&2
    exit 1
fi

###################################################################################################
# THRONE STEP 1. GENERATE RESPONSES USING EVALUATEE
###################################################################################################
WORKING_PATH="$ROOT/caches/${MODEL}_$DATASET"
mkdir -p $WORKING_PATH

if [ ! -f "$WORKING_PATH/responses/responses.json" ]; then
    PYTHONPATH=${pythonpath} torchrun --nproc_per_node 8 \
    throne_generate.py \
    --coco_file "$COCO_FILE" \
    --coco_image_dir "$COCO_IMAGE" \
    $COCO_SUB_FLAGS \
    --save_path "$WORKING_PATH/responses/responses.json" \
    $MODEL_GEN_FLAGS

fi

###################################################################################################
# THRONE STEP 2. PERFORM ABSTRACTIVE QA TO EVALUATE RESPONSES
###################################################################################################
for EVALUATOR in 'google/flan-t5-base' 'google/flan-t5-large' 'google/flan-t5-xl'; do
    if ! ls "$WORKING_PATH/evaluations/combined/evaluations_${EVALUATOR/\//_}.json"; then
        echo Evaluating with $EVALUATOR ...
        torchrun --nproc_per_node 8 throne_aqa_evaluation.py \
        --coco_file "$COCO_FILE" \
        --response_file "$WORKING_PATH/responses/responses.json" \
        --evaluator_model_path "$EVALUATOR" \
        --per_device_batch_size 32 \
        --save_path "$WORKING_PATH/evaluations/evaluations.json" \

    fi
done

###################################################################################################
# THRONE STEP 3. COMPUTE THRONE METRICS TO SCORE EVALUATEE
###################################################################################################
python throne_score_aqa.py \
--coco_val_ann_path $COCO_FILE \
--model_eval_path "$WORKING_PATH/evaluations/combined" \
--thresholds 5 8 9 \
--metric_strategy 'micro'

python throne_score_aqa.py \
--coco_val_ann_path $COCO_FILE \
--model_eval_path "$WORKING_PATH/evaluations/combined" \
--thresholds 5 8 9 \
--metric_strategy 'class'
