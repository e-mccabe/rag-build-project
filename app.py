import streamlit as st

from rag_build.loading import load_vault
from rag_build.chunking import chunk_all_documents
from rag_build.embedding import index_chunks
from rag_build.response import retrieval, generate_stream


st.set_page_config(page_title='Second Brain Assistant',page_icon='~',initial_sidebar_state='expanded')
st.title('Ask the second brain')

st.sidebar.title('Settings')
st.sidebar.write('Navigation Page')


@st.cache_resource
def build_index()-> int:

    documents = load_vault('data')
    chunks = chunk_all_documents(documents)
    index_chunks(chunks)
    return len(chunks)

n_chunks = build_index()
st.caption(f'Indexed {n_chunks} from the vault.')

if 'messages' not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt := st.chat_input('Ask something'):
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    
#    response = ask(prompt)['answer']
#    st.session_state.messages.append({'role':'assistant','content':response})
    with st.chat_message('assistant'):

        hits = retrieval(prompt)
        if not hits:
            answer = "I don't have anything in the corpus about that."
            st.markdown(answer)
            sources = []
        
        else:
            answer = st.write_stream(generate_stream(prompt,hits))
    
    st.session_state.messages.append({'role':'assistant','content':answer})

#        st.markdown(response)





