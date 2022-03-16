import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

bert_model = TFAutoModelForQuestionAnswering.from_pretrained('./QuestionAnswering/Fine_tune_BERT/')

def predictAnswer(context,question):
  inputs = tokenizer([context], [question], return_tensors="np")
  outputs = bert_model(inputs)
  start_position = tf.argmax(outputs.start_logits, axis=1)
  end_position = tf.argmax(outputs.end_logits, axis=1)
  answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
  return tokenizer.decode(answer)

