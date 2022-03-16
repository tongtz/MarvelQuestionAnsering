import streamlit as st
import os


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

if submit_button:
    st.markdown("test")


st.header("")
