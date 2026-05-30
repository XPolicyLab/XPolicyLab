#!/bin/bash

SPLIT="mmbench_dev_20230712"
LLM_MODEL_SIZE=2_8B
python -m llava_pythia.eval.model_vqa_mmbench \
    --model-path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llavaPhi-v0-3b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode pythia

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llavaPhi-v0-3b