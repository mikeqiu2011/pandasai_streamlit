from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai.smart_dataframe import SmartDataframe
import io
import seaborn as sns
import matplotlib

matplotlib.use(backend="TkAgg")

model = LocalLLM(
    api_base='http://localhost:11434/v1',
    model='llama3:8b'
)

st.title('Data analysis with PandasAI')

uploaded_file = st.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(2))

    df = SmartDataframe(data, config={'llm': model})
    prompt = st.text_area('Enter a question:')

    if st.button('Generate'):
        if prompt:
            with st.spinner('Generating...'):
                response = df.chat(prompt)
                st.write(response)