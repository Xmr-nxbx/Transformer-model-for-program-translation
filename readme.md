## Transformer model for program translation

### Introduction

Today, many studies focus on applying neural networks to software engineering tasks such as comment generation, code search, clone detection, and so on. Among them, the program translation task requires the model to translate the source code to the target code without changing its functionality. This task requires the model to understand the source code semantics and generate code based on the specifications of the target programming language.

This repository is created to investigate the program translation baseline Transformer. the CodeTrans dataset is shown in [CodeXGLUE/CodeTrans](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans).

In addition, the model has some feature, such as:

1. simple modification of parameters
2. gradient accumulation
3. `tf.function` acceleration
4. multi-GPU training

It should be noted that the gradient accumulation function is replicated in [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf).



### Dependency

- tensorflow 2
- tokenizers
- numpy
- tree-sitter

Besides, if evaluating the output, pycharm is required. (To be honest, my programming skills are limited)

### Project Composition

- `./data` folder is used to store datasets, vocabulary, references, model's ckpt, and predicted code.
- `./evaluator` folder holds the evaluation metrics. The evaluation metrics are from [CodeTrans](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator).
- `./network` and `./util` folders store the model and preprocessing files.

### Experimental settings
I believe you can see the `config` dict in `train.py`. Just change the value corresponding to the key listed in `config`. 

Note that the `"swap datasets by dictionary order": False` refers to translate the name of a programming language with a small dictionary order to another programming language.

### Usage

1. Save the dataset with the files like `keyword.file_name.language` to `./data/dataset_name/source/`. Where `keyword` in `[train, valid, test]`, `language` is the program language that the tree-sitter can parse.
2. run `prepare_data.py` to preprocess dataset.
3. run `train.py` to create the transformer model and generate output.
4. run `metric_eval.py` to evaluate the output in terms of BLEU, EM, CodeBLEU metrics. 

Where step 4 needs to be run in pycharm, select the folder `evaluator/CodeBLEU` and mark directory as sources root.

### Result

My lab server specs are i9 9900k and RTX2080Ti, but I haven't finished training yet.

#### Other

My research is in program translation, and I hope I can graduate successfully.

深度学习太卷了，菜鸡逐渐失去梦想...

