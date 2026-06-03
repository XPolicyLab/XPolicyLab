#!/bin/bash
LLM_MODEL_SIZE=2_8B
python -m llava_pythia.eval.model_vqa_loader \
    --model-path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data/team/zhumj/data/finetune/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode pythia

python -m llava_pythia.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llavaPhi-v0-3b.jsonl
