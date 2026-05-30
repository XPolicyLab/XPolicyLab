#!/bin/bash
LLM_MODEL_SIZE=2_8B
python -m llava_pythia.eval.model_vqa_science \
    --model-path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode pythia

python llava_pythia/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b_result.json

