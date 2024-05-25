import os
import argparse


def generate_command_codesearch(task_type, model_type, pruning_strategy, pruning_ratio):
    data_path = f"./data/{task_type}"
    if pruning_strategy=='None':
        model_dir = f"./models/{task_type}/{model_type}/base"
        seq_len=200
    else:
        model_dir = f"./models/{task_type}/{model_type}/{pruning_strategy}/{pruning_ratio}"
        seq_lens={10:180,20:160,30:140,40:120,50:100}
        seq_len=seq_lens[pruning_ratio]
    if model_type=='codebert':
        model_type_='roberta'
        model_path='microsoft/codebert-base'
    else:
        model_type_='codet5'
        model_path = 'Salesforce/codet5-base'
        # model_path='/home/dell/dl/DietCode-master/dietCodeBERT/codesearch/codet5-base'
    train_cmd = f"python3 {task_type}/run_classifier.py --model_type {model_type_} --tokenizer_name {model_path} --model_name_or_path {model_path} --task_name codesearch --do_train --do_eval --prune_strategy {pruning_strategy} --output_dir {model_dir} --data_dir {data_path}/train_valid/java --train_file train.txt --dev_file valid.txt --max_seq_length {seq_len} --per_gpu_train_batch_size 256 --per_gpu_eval_batch_size 256 --learning_rate 1e-5 --num_train_epochs 4  --lang java --gradient_accumulation_steps 1 --overwrite_output_dir"
    eval_cmd = f"python3 {task_type}/run_classifier.py --model_type {model_type_} --tokenizer_name {model_path} --model_name_or_path {model_path} --task_name codesearch --do_predict --prune_strategy {pruning_strategy} --output_dir {model_dir} --pred_model_dir {model_dir}/checkpoint-best --test_result_dir {model_dir}/0_batch_result.txt --data_dir {data_path}/test/java --test_file batch_0.txt --max_seq_length {seq_len} --per_gpu_eval_batch_size 512 --learning_rate 1e-5"
    test_cmd = f"python3 {task_type}/mrr.py --test_dir {model_dir}/0_batch_result.txt"

    return train_cmd,eval_cmd,test_cmd

def generate_command_code2nl_codebert(task_type, model_type, pruning_strategy, pruning_ratio):
    data_path = f"./data/{task_type}"
    if pruning_strategy=='None':
        model_dir = f"./models/{task_type}/{model_type}/base"
        seq_len=256
    else:
        model_dir = f"./models/{task_type}/{model_type}/{pruning_strategy}/{pruning_ratio}"
        seq_lens={10:230,20:205,30:180,40:154,50:128}
        seq_len=seq_lens[pruning_ratio]
    model_type_ = 'roberta'
    model_path = 'microsoft/codebert-base'
    train_cmd = f"python code2nl/CodeBERT/run.py --model_type {model_type_} --tokenizer_name {model_path} --model_name_or_path {model_path} --do_train --do_eval --prune_strategy {pruning_strategy} --train_filename {data_path}/CodeSearchNet/java/train.jsonl --dev_filename {data_path}/CodeSearchNet/java/valid.jsonl --output_dir {model_dir} --max_source_length {seq_len} --max_target_length 128 --beam_size 10 --train_batch_size 384 --eval_batch_size 384 --learning_rate 5e-5 --num_train_epochs 15"

    # 测试命令
    test_cmd = f"python code2nl/CodeBERT/run.py --model_type {model_type_} --tokenizer_name {model_path} --model_name_or_path {model_path} --do_test --prune_strategy {pruning_strategy} --test_filename {data_path}/CodeSearchNet/java/test.jsonl --max_source_length {seq_len} --output_dir {model_dir} --max_target_length 128 --beam_size 10 --load_model_path {model_dir}/checkpoint-best-bleu/pytorch_model.bin --eval_batch_size 512"

    return train_cmd,test_cmd

def generate_command_code2nl_codet5(task_type, model_type, pruning_strategy, pruning_ratio):
    data_path = f"./data/{task_type}"
    if pruning_strategy=='None':
        model_dir = f"./models/{task_type}/{model_type}/base"
        seq_len=256
    else:
        model_dir = f"./models/{task_type}/{model_type}/{pruning_strategy}/{pruning_ratio}"
        seq_lens={10:230,20:205,30:180,40:154,50:128}
        seq_len=seq_lens[pruning_ratio]
    model_path = 'Salesforce/codet5-base'
    train_eval_cmd = f"python code2nl/CodeT5/run_gen.py --model_type codet5 --task summarize --sub_task java --tokenizer_name {model_path} --model_name_or_path {model_path} --do_train --do_eval --do_eval_bleu --do_test --prune_strategy {pruning_strategy} --data_num -1 --num_train_epochs 8 --warmup_steps 1000 --learning_rate 5e-5 --patience 2 --data_dir {data_path}/CodeSearchNet/java --cache_path {model_dir}/cache_data --output_dir {model_dir} --save_last_checkpoints --always_save_model --res_dir {model_dir}/prediction --res_fn {model_dir}/result.txt --train_batch_size 96 --eval_batch_size 96 --max_source_length {seq_len} --max_target_length 128 --summary_dir {model_dir}/tensorboard"
    return train_eval_cmd


def main():
    parser = argparse.ArgumentParser(description="Generate command based on parameters.")

    parser.add_argument("--task_type", type=str, help="Type of the task",choices=['codesearch', 'code2nl'])
    parser.add_argument("--model_type", type=str, help="Type of the model",choices=['codebert', 'codet5'])
    parser.add_argument("--prune_strategy", type=str, help="Pruning strategy",choices=['leancode', 'dietcode', 'leancode_d','None'])
    parser.add_argument("--prune_ratio", type=int, help="Pruning ratio",choices=[10,20,30,40,50])

    args = parser.parse_args()

    if args.task_type=='codesearch':
        train_cmd,eval_cmd,test_cmd = generate_command_codesearch(args.task_type, args.model_type, args.prune_strategy, args.prune_ratio)
        print('============================Training cmd==========================')
        print(train_cmd)
        print('============================Evaluating cmd==========================')
        print(eval_cmd)
        print(test_cmd)
    elif args.model_type=="codebert":
        assert args.task_type=='code2nl'
        train_cmd,test_cmd=generate_command_code2nl_codebert(args.task_type, args.model_type, args.prune_strategy, args.prune_ratio)
        print('============================Training cmd==========================')
        print(train_cmd)
        print('============================Testing cmd==========================')
        print(test_cmd)
    else:
        assert args.task_type=='code2nl'
        assert args.model_type=='codet5'
        train_and_test_cmd=generate_command_code2nl_codet5(args.task_type, args.model_type, args.prune_strategy, args.prune_ratio)
        print('============================Training and test cmd==========================')
        print(train_and_test_cmd)


if __name__ == "__main__":
    main()
