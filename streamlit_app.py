import streamlit as st

st.title('StarRec')

huggingface_api_key = st.sidebar.text_input('Hugging Face API Key', type='password')

with st.form('test_form'):
    text = st.text_area('Enter text:', 'What\'s new about U.S. presidential election?')
    submitted = st.form_submit_button('Submit')
    if not huggingface_api_key.startswith('hf-'):
        st.info("hahahahahahahahaha")