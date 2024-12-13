import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains.conversation.memory import ConversationStringBufferMemory, ConversationBufferWindowMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent
import pickle
from langchain.agents import Tool
from langchain import hub
from langchain.agents import AgentExecutor

with open("test", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
df_document = b


st.title("💬 Pharmacogenomics Bot")
st.write(
    "This is a simple chatbot that is trained on documents that were queried from the CPIC database."
    "It is trained on CYP2C19 genes so far, and can understand phenotypes such as *17/*17 or ultra-rapid metabolizer."
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# API Key
#OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Initialize Chroma VectorStore
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = Chroma.from_documents(
            documents=df_document,
            collection_name="my_collection",
            embedding=embeddings,
            persist_directory="./chroma_db"  # In-memory mode
        )

    vector_store = st.session_state.vector_store
    # Create an OpenAI client.
    # client = OpenAI(api_key=openai_api_key)
    #
    # # Create a session state variable to store the chat messages. This ensures that the
    # # messages persist across reruns.
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []
    #
    # # Display the existing chat messages via `st.chat_message`.
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])
    #
    # # Create a chat input field to allow the user to enter a message. This will display
    # # automatically at the bottom of the page.
    # if prompt := st.chat_input("How can I help? 🤖 "):
    #
    #     # Store and display the current prompt.
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #
    #     # Generate a response using the OpenAI API.
    #     stream = []
    #     for m in st.session_state.messages:
    #         stream.append(generate_response(openai_api_key, m["content"]))
    #
    #     # Stream the response to the chat using `st.write_stream`, then store it in
    #     # session state.
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(stream)
    #     st.session_state.messages.append({"role": "assistant", "content": response})

# Query Input
    query = st.text_input("Ask your question:")

    if st.button("Submit"):
        if query:
        # Set up QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=openai_api_key),
                retriever=vector_store.as_retriever()
            )
            answer = qa_chain.run(query)
            st.write("Answer:", answer)
