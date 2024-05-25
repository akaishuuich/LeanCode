#!/bin/bash

WORKDIR="your_LeanCode_path/LeanCode"
export PYTHONPATH=$WORKDIR

python codesearch/run_classifier.py --model_type roberta --task_name codesearch --gen_weight --data_dir ./data/codesearch/train_valid/java --train_file train.txt  --tokenizer_name microsoft/codebert-base --model_name_or_path microsoft/codebert-base  --pred_model_dir ./models/codesearch/codebert/base/checkpoint-best --max_seq_length 200 --per_gpu_train_batch_size 256 --per_gpu_eval_batch_size 256  --lang java --gradient_accumulation_steps 1  --prune_strategy None  --output_dir ./models/codesearch/codebert/base

python codesearch/run_classifier.py --model_type codet5 --task_name codesearch --gen_weight --data_dir ./data/codesearch/train_valid/java --train_file train.txt  --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base  --pred_model_dir ./models/codesearch/codet5/base/checkpoint-best --max_seq_length 200 --per_gpu_train_batch_size 256 --per_gpu_eval_batch_size 256  --lang java --gradient_accumulation_steps 1  --prune_strategy None  --output_dir ./models/codesearch/codet5/base

python code2nl/CodeT5/run_gen.py --model_type codet5 --gen_weight  --tokenizer_name  Salesforce/codet5-base --model_name_or_path  Salesforce/codet5-base --task summarize --sub_task java  --data_num -1   --data_dir ./data/code2nl/CodeSearchNet/java --cache_path  ./models/code2nl/codet5/base/cache_data --output_dir ./models/code2nl/codet5/base --summary_dir ./models/code2nl/codet5/base/tensorboard    --eval_batch_size 128 --max_source_length 256 --max_target_length 128 --res_dir ./models/code2nl/codet5/base/prediction --load_model_path ./models/code2nl/codet5/base/checkpoint-best-bleu/pytorch_model.bin --prune_strategy None

python code2nl/CodeBERT/run.py --model_type roberta --tokenizer_name microsoft/codebert-base --model_name_or_path microsoft/codebert-base --gen_weight  --prune_strategy None --train_filename ./data/code2nl/CodeSearchNet/java/train.jsonl  --output_dir ./models/code2nl/codebert/base --max_source_length 256 --max_target_length 128 --beam_size 10  --eval_batch_size 384 --load_model_path ./models/code2nl/codebert/base/checkpoint-best-bleu/pytorch_model.bin

python utils/gen_weight_dic.py



