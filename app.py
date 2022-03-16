import streamlit as st
import os
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering

st.set_page_config(
    page_title="Question Answering",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

st.title("🔑 Question Answering System")
st.header("")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   some introducation
-   second one
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 Paste document ")


with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
    
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)",],
        )


    with c2:
        
        context = st.text_area(
            "Paste your text below",
            height=300,
        )

        question = st.text_area(
            "Paste your question below ",
            height=100,
        )
        
        submit_button = st.form_submit_button(label="✨ Get me the answer!")


st.markdown("## 🎈 Check answer ")

st.header("")

model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

bert_model = TFAutoModelForQuestionAnswering.from_pretrained('./Fine_tune_BERT/')

def predict(context,question):
  inputs = tokenizer([context], [question], return_tensors="np")
  outputs = bert_model(inputs)
  start_position = tf.argmax(outputs.start_logits, axis=1)
  end_position = tf.argmax(outputs.end_logits, axis=1)
  answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
  return tokenizer.decode(answer)

if submit_button:
	st.markdown(predict(context,question))


st.header("")
