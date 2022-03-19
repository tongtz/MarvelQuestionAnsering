# NLP: Question Answering System based on SubjQA

Team Members: Wade Wei, Tong Zhou.

Objective: create an NLP system that provides a single answer to a question, which is obtained by selecting a span of text from the context.

This repository contains the essential code in order to fine-tune BERT on the [Subjqa](https://huggingface.co/datasets/subjqa) dataset. Additionally, the technique of Knowledge Distillation is applied by fine-tuning [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the dataset using BERT as the teacher model. 

## 1. What is Subjqa?
SubjQA is a question answering dataset that focuses on subjective (as opposed to factual) questions and answers. The dataset consists of roughly 10,000 questions over reviews from 6 different domains: books, movies, grocery, electronics, TripAdvisor (i.e. hotels), and restaurants. Each question is paired with a review and a span is highlighted as the answer to the question

## 2. GitHub Framework
```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── app.py                  <- app to run project / user interface (streamlit)
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── data_processing.py  <- script to preprocess data
    ├── train_regression.py <- script to train logistice regression model
    ├── train_BERT.py       <- script to train model DistilBERT model. This will also save the model within '/DistilBERT'
    ├── evaluation.py       <- script to evaluate the model
    ├── predict.py          <- script to make generate answer from context and question
├── presentation            <- project presentation
├── fine_tune_BERT          <- directory for trained models
    ├── tf_model.h5         <- pretrained model 
    ├── config.json.        <- pretrained model 
├── .gitignore              <- git ignore file
├── .gitattributes          <- git attributes file
├── .DS_Store               <- DS_Store file
```

## 3. How to run?

* To installation: To quickly use our modified version of QuestionAnswering, clone this repository and install the necessary requirements by running ```
pip install -r requirements.txt```
* To train a logistic regression on our dataset: please run ```python -m scripts.train_regression```
* To train and fine-tune BERT: please run ```python -m scripts.train_BERT```. This script will automatically train a DistilBERT model and save the fine-tuned BERT model in the same directory.
* To evaluate the fine-tuned BERT model: please run ```python -m scripts.evaluation```. Note that you need to load a model to do evaluation, whether its from your trained model by running our ```train_BERT.py```, or from our provided model. Please make sure to change directory if you want to use provided model.
* After trained and evaluate the model, run and execute the streamlit app for applications demo:```streamlit run app.py```. This app will use default pretrained model stored in ```/fine_tune_BERT```, if you want to use the model you treained, please update the path here: ```bert_model = TFAutoModelForQuestionAnswering.from_pretrained('./fine_tune_BERT/')```.

## 4. Code and Paper References

* A part of the code has been based on the publicly available code of [🤗 HuggingFace Transformers library](https://huggingface.co/models?library=transformers) and their corresponding research project on [DistilBERT](https://huggingface.co/distilbert-base-uncased).
