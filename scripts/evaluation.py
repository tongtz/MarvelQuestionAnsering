from scripts.data_processing import pipeline
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf
from tqdm import tqdm
val_dataset = pipeline('validation')
test_dataset = pipeline('test')

bert_model = TFAutoModelForQuestionAnswering.from_pretrained('./DistilBERT/')
model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


TP, FP, FN = 0, 0, 0

model = bert_model

for item in tqdm(val_dataset):
  context = item['context']
  question = item['question']
  true_answer = item['answers']['text'][0]

  inputs = tokenizer([context], [question], return_tensors="np")
  outputs = model(inputs)
  start_position = tf.argmax(outputs.start_logits, axis=1)
  end_position = tf.argmax(outputs.end_logits, axis=1)

  answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
  predicted_answer = tokenizer.decode(answer)

  #print(context)
  #print(question)

  #print(true_answer, '|', predicted_answer)

  if true_answer in predicted_answer or predicted_answer in true_answer:
    TP += 1
  if len(predicted_answer) < 1:
    FN += 1
  if true_answer not in predicted_answer and predicted_answer not in true_answer:
    FP += 1

  #print(TP, FP, P)

print('DistilBERT val sensitivity ', TP/(TP + FN))
#print('DistilBERT val precision ', TP/(TP + FP))
print('DistilBERT val F1 score ', 2 * TP/(2*TP + FP + FN))

print('')
TP, FP, FN = 0, 0, 0

#model = bert_model

for item in tqdm(test_dataset):
  context = item['context']
  question = item['question']
  true_answer = item['answers']['text'][0]

  inputs = tokenizer([context], [question], return_tensors="np")
  outputs = model(inputs)
  start_position = tf.argmax(outputs.start_logits, axis=1)
  end_position = tf.argmax(outputs.end_logits, axis=1)

  answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
  predicted_answer = tokenizer.decode(answer)

  #print(true_answer, '|', predicted_answer)

  if true_answer in predicted_answer or predicted_answer in true_answer:
    TP += 1
  if len(predicted_answer) < 1:
    FN += 1
  if true_answer not in predicted_answer and predicted_answer not in true_answer:
    FP += 1

  #print(TP, FP, P)

print('DistilBERT test sensitivity ', TP/(TP + FN))
print('DistilBERT test F1 score ', 2 * TP/(2*TP + FP + FN))
