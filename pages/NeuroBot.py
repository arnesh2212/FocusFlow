import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
google_api_key = "AIzaSyAfNuOsINiHgzSBPdFJaSC_UqoIV35tiJE"
os.environ["GOOGLE_API_KEY"] = google_api_key
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
st.set_page_config(layout= "centered")

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
import io

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/gradient-purple-colorful-background-modern-wave-abstract_343694-2280.jpg?t=st=1724554396~exp=1724557996~hmac=4de036af7b28c4cfd3435c9de395c3551388a7141c8733029c729522e251af6e&w=1060");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


import os
import re

def load_txt(file_path):
    """
    Reads the text content from a TXT file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the TXT file.

    Returns:
    - str: The text content of the TXT file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text.strip()

# Replace the path with your file path
pdf_text = load_txt(file_path="Neuro.txt")

# Retrieve API key from environment variable
google_api_key = "AIzaSyAfNuOsINiHgzSBPdFJaSC_UqoIV35tiJE"
os.environ["GOOGLE_API_KEY"] = google_api_key

# Check if the API key is available
if google_api_key is None:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# # File Upload with user-defined name




# # Create Context
# context = pdf_text

# # Split Texts
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
# texts = text_splitter.split_text(context)

# # Chroma Embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

# # Get User Question

def question_answering(user_question):
    # Get Relevant Documents
    docs = vector_index.get_relevant_documents(user_question)
    prompt_template = """
    You are a chatbot to help neurodiverse students study better, so based on the given information answer the following question, you mainly cater to indian students\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # Create Prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Load QA Chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, api_key=google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Get Response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response
# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

st.write('# Chat with NeuroPal')

# Chat history (allows to ask multiple questions)
# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = [
        dict(
            role=MODEL_ROLE,
            content="I am NeuroPal, your companion to help you study better. How can I help you?",
            avatar=AI_AVATAR_ICON,
        )
    ]
    st.session_state.gemini_history = []
    print('new_cache made')
st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)

# Display chat messages from history on app rerun2
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Your message here...'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content= prompt,
        )
    )
    ## Send message to AI
    #rea = question_answering(prompt)
    response = st.session_state.chat.send_message(
        "You are a chatbot to help neurodiverse students study better, So answer the following question accordingly, Our app also has other features like a Pomodjsonoro timer, To-Do list, and a attention Monitor , which also can be recommended to the user if required." + prompt,
        stream=True,
    )
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        assistant_response = response
        # Streams in a chunk at a time
        for chunk in response:
            # Simulate stream of chunk
            # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
            for ch in chunk.text.split(' '):
                full_response += ch + ' '
                time.sleep(0.05)
                # Rewrites with a cursor at end
                message_placeholder.write(full_response + '▌')
        # Write full message with placeholder
        message_placeholder.write(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )