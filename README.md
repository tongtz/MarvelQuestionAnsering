# Question Answering System based on SubjQA

## 1. What is Subjqa?
This repository contains the essential code in order to fine-tune BERT on the Subjqa dataset. Additionally, the technique of Knowledge Distillation is applied by fine-tuning DistilBERT on the dataset using BERT as the teacher model. 

## 2. How to run?

* To installation: To quickly use our modified version of QuestionAnswering, clone this repository and install the necessary requirements by running ```
pip install -r requirements.txt```
* To fine-tune BERT: please run ``` .ipynb```. This notebook will automatically save the fine-tuned BERT model in ```./Fine_tune_BERT/.```
* To evaluate the fine-tuned BERT model: please run ``` .ipynb```.

## 3. Code and Paper References

* A part of the code has been based on the publicly available code of 🤗 HuggingFace Transformers library and their corresponding research project on DistilBERT (code).
