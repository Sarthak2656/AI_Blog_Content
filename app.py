import streamlit as st
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
st.set_page_config(page_title="Generate Blogs",
                   page_icon='blog.png',
                   layout='centered',
                   initial_sidebar_state='collapsed')

def getLLamaresponse(input_text, no_words, blog_style):
    
    llm = AutoModelForCausalLM.from_pretrained(
        'models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        temperature= 0.03,
        max_new_tokens=500
    )


    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'], template=template)


    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response


def add_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #030101;
            font-family: 'Calibri', sans-serif;
        }
        .stHeader {
            color: #4CAF50;
            text-align: center;
            font-size: 32px;
            margin-bottom: 20px;
        }
        .stTextInput, .stSelectbox, .stButton {
            margin: 10px 0;
        }
        .stTextInput input, .stSelectbox select {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .stButton button {
            background-color: #4d4f4e;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            border-radius: 4px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #464d47;
        }
        .stColumns {
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# Add CSS to the app
add_custom_css()

st.header("📓 Generate Blogs 📓")

input_text = st.text_input("Enter the Blog Topic", key='blog_topic')
col1, col2 = st.columns([5, 5], gap="medium")

with col1:
    no_words = st.text_input('Number of Words', key='no_words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People','Teachers'), index=0, key='blog_style')

submit = st.button("Generate", key='generate_button')

# Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
