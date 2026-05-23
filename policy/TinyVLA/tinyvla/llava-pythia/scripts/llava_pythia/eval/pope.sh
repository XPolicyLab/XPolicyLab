#!/bin/bash
LLM_MODEL_SIZE=2_8B
python -m llava_pythia.eval.model_vqa_loader \
    --model-path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /data/team/zhumj/data/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode pythia

python llava_pythia/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llavaPhi-v0-3b.jsonl