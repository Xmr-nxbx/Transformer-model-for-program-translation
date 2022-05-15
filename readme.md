# Transformer model for program translation

## Introduction

Today, many studies focus on applying neural networks to software engineering tasks such as comment generation, code search, clone detection, and so on. Among them, the program translation task requires the model to translate the source code to the target code without changing its functionality. This task requires the model to understand the source code semantics and generate code based on the specifications of the target programming language.

This repository is created to investigate the program translation baseline Transformer. The CodeTrans dataset is shown in [CodeXGLUE/CodeTrans](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans).

In addition, our model has some feature, such as:

1. simple modification of parameters
2. gradient accumulation
3. `tf.function` acceleration
4. multi-GPU training
5. mixed precision (float16 and float32)

It should be noted that the gradient accumulation function is copied from [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf).

## Dependency

- tensorflow 2
- tokenizers
- numpy
- tree-sitter

Besides, if evaluating the output, pycharm is required. (To be honest, my programming skills are limited)

## Project Composition

- `./data` folder is used to store datasets, vocabulary, references, model's ckpt, and predicted code.
- `./evaluator` folder holds the evaluation metrics. The evaluation metrics are from [CodeTrans](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator).
- `./network` and `./util` folders store the model and preprocessing files.

## Experimental settings
I believe you can see the `config` dict in `train.py`. Just change the value corresponding to the key listed in `config`. 

Note that the `"swap datasets by dictionary order": False` refers to translate the name of a programming language with a small dictionary order to another programming language.

## Usage

1. Save the dataset with the files like `keyword.file_name.language` to `./data/dataset_name/source/`. Where `keyword` in `[train, valid, test]`, `language` is the program language that the tree-sitter can parse.
2. run `prepare_data.py` to preprocess dataset.
3. run `train.py` to create the transformer model and generate output.
4. run `metric_eval.py` to evaluate the output in terms of BLEU, EM, CodeBLEU metrics. 

Where step 4 needs to be run in pycharm, select the folder `evaluator/CodeBLEU` and mark directory as sources root.

## Result

Note that I did not set up warmup because of the high learning rate with few training steps.

**Java to C#**

| model | layer | hidden | learning rate | BLEU | Exact Match | CodeBLEU |
|---|---|---|---|---|---|---|
|Transformer-baseline| 12 | 768 | - | 55.84 | 33.0 | 63.74 |
|Transformer| 12 | 768 | 1e-4 | 50.64 | 31.3 | 58.24 |
|Transformer| 12 | 768 | 5e-5 | 53.01 | 35.2 | 60.98 |

**C# to Java**

| model | layer | hidden | learning rate | BLEU | Exact Match | CodeBLEU |
|---|---|---|---|---|---|---|
|Transformer-baseline| 12 | 768 | - | 50.47 | 37.9 | 61.59 |
|Transformer| 12 | 768 | 1e-4 | 45.01 | 31.4 | 53.06 |
|Transformer| 12 | 768 | 5e-5 | 45.91 | 33.0 | 53.89 |

#### Other

My research is in program translation, and I hope I can graduate successfully.

