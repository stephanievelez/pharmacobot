**Interactive Agent with Streamlit**
This repository contains the components for creating an interactive agent that utilizes pre-processed data stored in a pickle file. The agent is designed to provide intelligent and context-aware responses, leveraging a user-friendly interface powered by Streamlit.

**Overview
Data Preparation**

The project starts with a pickle file that contains data structured as a DataFrame.
The DataFrame is converted into a collection of documents, which forms the foundation for the agent's interactions and retrieval capabilities.

**Agent Interaction**

The agent processes user queries by leveraging its document collection and maintains context to respond intelligently based on previous queries.
This ensures a dynamic and conversational experience tailored to the user's needs.

**Streamlit Integration**

Streamlit, an open-source Python library, is used to build an interactive interface for the agent.
Users can interact with the agent directly through a clean and responsive web application.

**Data Setup**

Ensure the pickle file with the processed data is present in the project directory.

**Running the Application**
Install the required dependencies:
Copy code
pip install streamlit  
Launch the Streamlit app:
streamlit run <your_app_name>.py  
