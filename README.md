# Question Answering System based on SubjQA

## 1. What is Subjqa?
This repository contains the essential code in order to fine-tune BERT on the Subjqa dataset. Additionally, the technique of Knowledge Distillation is applied by fine-tuning DistilBERT on the dataset using BERT as the teacher model. 

## 2. How to run?

* To installation: To quickly use our modified version of QuestionAnswering, clone this repository and install the necessary requirements by running ```
pip install -r requirements.txt```
* To train a logistic regression on our dataset: please run ```python -m scripts.train_regression```
* To train and fine-tune BERT: please run ```python -m scripts.train_BERT```. This script will automatically train a DistilBERT model and save the fine-tuned BERT model in the same directory.
* To evaluate the fine-tuned BERT model: please run ```python -m scripts.evaluation```. Note that you need to load a model to do evaluation, whether its from your trained model by running our ```train_BERT.py```, or from our provided model. Please make sure to change directory if you want to use provided model.

## 3. Code and Paper References

* A part of the code has been based on the publicly available code of 🤗 HuggingFace Transformers library and their corresponding research project on DistilBERT (code).
