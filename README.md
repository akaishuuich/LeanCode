# LeanCode
This repo provides the code for reproducing the experiments in LeanCode. LeanCode is a novel program simplification approach that utilize code contexts and attention scores for representing the importance levels of tokens.
## Requirements
- [python3](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/get-started/locally/)
## QuickStart
### Prepare data
We used two datasets from [CodeBERT](https://arxiv.org/pdf/2002.08155): code search and code summarization datasets. These are the extensions of [CodeSearchNet](https://github.com/github/CodeSearchNet).
You can download the code search dataset from [this website](https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo), and download the code summarization dataset from [this website](https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h) Or use the following command.

```
mkdir data data/codesearch data/code2nl
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo
unzip codesearch_data.zip
rm  codesearch_data.zip
cd ../../codesearch
python process_data.py

cd ../data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../..
```
### Reproduce the experimental results
We conducted code simplification experiments on two tasks, `code search` and `code summarization`, using two models: `CodeBERT` and `CodeT5`. These experiments involved three code simplification strategies—`LeanCode`, `DietCode`, and `LeanCode with the removal algorithm of DietCode` and five code simplification ratios: `10%`, `20%`, `30%`, `40%`, and `50%`.  
  
You can use `gen_cmd.py` in `sh` floder to get the training and evaluating cmd to run a broad set of experiments by simply passing the `task_type`, `model_type`, `prune_strategy` and `prune_ratio` arguments. 
Below is a table listing each parameter along with its description and available options:

| Parameter          | Description                                           | Options                                                         |
|--------------------|-------------------------------------------------------|-----------------------------------------------------------------|
| **task_type**      | Specifies the type of task to perform.                | `codesearch`, `code2nl`                                         |
| **model_type**     | Determines the model used for the task.               | `codebert`, `codet5`                                            |
| **prune_strategy** | Defines the pruning strategy to enhance model performance. | `leancode`, `dietcode`, `leancode_d`, `None`                  |
| **prune_ratio**    | Specifies the percentage of the model to prune.       | `10`, `20`, `30`, `40`, `50`                                    |

#### Usage Examples
for example , If you want to perform a code search task using the CodeBERT model with a LeanCode strategy at a 40% pruning ratio. execute the command as shown below ，then you can obtain the corresponding training and evaluation commands, which then allow you to conduct the respective experiments

```
python sh/gen_cmd.py --task_type codesearch --model_type codebert --prune_strategy leancode --prune_ratio 40
```
If you need to conduct an experiment on CodeBERT for code search without pruning (base expirement), you can use the following commands to obtain the necessary commands for training and inference.
```
python sh/gen_cmd.py --task_type codesearch --model_type codebert --prune_strategy None
```
### Generate the weight_dic of LeanCode
If you want to generate the weight dictionaries by yourself , Go to `sh` folder, set the `WORKDIR` in `gen_weights.sh` to be your cloned LeanCode repository path. using the following command you can get the weights and get it saved. Ensure you have completed the base experiments and saved the fine-tuned model; the commands provided in the previous section will assist you in this process.
```
bash sh/gen_weights.sh
```
### Other works
- the `analyse_attention` folder can help you analyse self_attention , self attention of cls token and encoder-decoder attention with the example of BubbleSort.  
- the `alalyse_weight.py` in `utils` folder can help you get Statistics of cls attention scores and encoder-decoder attention scores based on the training dataset .
