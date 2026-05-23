#!/bin/bash
LLM_MODEL_SIZE=2_8B
python -m llava_pythia.eval.model_vqa_loader \
    --model-path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode pythia

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llavaPhi-v0-3b

cd eval_tool

python calculation.py --results_dir answers/llavaPhi-v0-3b
