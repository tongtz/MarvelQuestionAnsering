import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset, concatenate_datasets
import tensorflow_hub as hub

from tqdm import tqdm

import nltk
nltk.download('punkt')
from nltk import tokenize

from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def pipeline(split):

    train_dataset1 = load_dataset("subjqa", "books", split=split)
    train_dataset2 = load_dataset("subjqa", "electronics", split=split)
    train_dataset3 = load_dataset("subjqa", "grocery", split=split)
    train_dataset4 = load_dataset("subjqa", "movies", split=split)
    train_dataset5 = load_dataset("subjqa", "restaurants", split=split)
    train_dataset6 = load_dataset("subjqa", "tripadvisor", split=split)

    train_dataset = concatenate_datasets([train_dataset1,
                                          train_dataset2,
                                          train_dataset3,
                                          train_dataset4,
                                          train_dataset5,
                                          train_dataset6])

    train_dataset = train_dataset.filter(lambda example: len(example["answers"]["text"]) > 0)

    print(train_dataset)

    return train_dataset

def create_feature(train_dataset):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print("module %s loaded" % module_url)

    train_cos_sim_list = []
    train_euc_dis_list = []
    target_list = []

    print("*****start tokenizing the dataset*****")

    for val in tqdm(train_dataset):
        context = tokenize.sent_tokenize(val["context"])
        question = val["question"]
        embedded_context = []
        # print(context)
        for i in range(len(context)):
            if str(val["answers"]["text"][0]) in str(context[i]):
                # print(str(val["answers"]["text"])[2:-2])
                # print(i)
                target_list.append(i)
                break
        else:
            target_list.append(0)  # assume it's idx 0
        for item in context:
            embedded_context.append(model([item])[0].numpy())

        # print(len(embedded_context), context)
        embed_question = model([question])[0]
        cos_sim = []
        euc_dis = []
        for item in embedded_context:
            cos_sim.append(1 - distance.cosine(item, embed_question))
            euc_dis.append(distance.euclidean(item, embed_question))
        # print(len(cos_sim), len(euc_dis))

        train_cos_sim_list.append(cos_sim)
        train_euc_dis_list.append(euc_dis)

    return train_cos_sim_list, train_euc_dis_list, target_list

def create_data_df(train_euc_dis_list, target_list):
    data = pd.DataFrame({'euc_dis': train_euc_dis_list,
                         'target': target_list})
    data = data[data["euc_dis"].apply(lambda x: len(x)) < 50].reset_index(drop=True)
    return data

def create_features(data):
    train = pd.DataFrame()

    for k in range(len(data["euc_dis"])):
        dis = data["euc_dis"][k]
        for i in range(len(dis)):
            train.loc[k, "column_euc_" + "%s" % i] = dis[i]

    print("Finished")
    train["target"] = data["target"]
    return train

def prepare_reg_features(split):
    train_dataset = pipeline(split)
    train_cos_sim_list, train_euc_dis_list, target_list = create_feature(train_dataset)
    data = create_data_df(train_euc_dis_list, target_list)
    train = create_features(data)
    train = train.fillna(0)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(train.iloc[:, :-1])

    train_x, test_x, train_y, test_y = train_test_split(X, train.iloc[:, -1], train_size=0.8, random_state=5)
    return train_x, test_x, train_y, test_y







