#!/bin/bash
LLM_MODEL_SIZE=2_8B
python -m llava_pythia.eval.model_vqa \
    --model-path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode pythia

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llavaPhi-v0-3b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llavaPhi-v0-3b.json

