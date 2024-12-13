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
            persist_directory="./chroma_db",  # Where to save data locally, remove if not neccesary
        )  # In-memory mode

    vector_store = st.session_state.vector_store
    vector_store.persist()
    # Create an OpenAI client.


# Query Input
    query = st.text_input("Ask your question:")
    if st.button("Submit"):
        if query:
        # Set up QA chain
            llm = OpenAI(openai_api_key=openai_api_key, temperature = 0.0)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_store.as_retriever()
            )

            conversational_memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=4,  # Number of messages stored in memory
                return_messages=True  # Must return the messages in the response.
            )

            tools = [
                Tool(
                    name='CPIC Bot',
                    func=qa.invoke,
                    description=(
                    """use this tool when answering how to dose a medication given a phenotype and or a diplotype"""
                    )
                )
            ]

            prompt = hub.pull("hwchase17/react-chat")
            agent = create_react_agent(
                tools=tools,
                llm=llm,
                prompt=prompt,
            )

            agent_executor = AgentExecutor(agent=agent,
                                       tools=tools,
                                       verbose=True,
                                       memory=conversational_memory,
                                       max_iterations=30,
                                       max_execution_time=600,
                                       handle_parsing_errors=True
                                       )

            answer = agent_executor.invoke({"input": query})



            #answer = qa_chain.invoke({"input": query})
            st.write("Answer:", answer['output'])
