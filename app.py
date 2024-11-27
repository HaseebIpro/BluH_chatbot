import streamlit as st
from utils import handle_user_query

def main():

    left, right = st.columns([1.0, 0.19], vertical_alignment="bottom")

    left.title('Ask BluH')

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Pass your prompt here')
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})
        response = handle_user_query(prompt, st.session_state.messages)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})

    if right.button('Clear Chat'):
        st.session_state.messages = []
        
    



if __name__ == "__main__":
    main()