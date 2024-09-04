from pyvis.network import Network
from neo4j import GraphDatabase
import streamlit as st
import os
from dotenv import load_dotenv
from neo4j.exceptions import SessionExpired  # Import the SessionExpired exception
# import time  # Add this import
from langsmith import traceable
from langsmith.wrappers import wrap_openai

load_dotenv()
# Setup connection to Neo4j database
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
groq_api_key = os.getenv("groq_api_key")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY2')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
#LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Setup llm
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
def initialize_llm(api_key=None, use_openai=True):
    if use_openai:
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    else:
        return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Python object for the connected instance
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# Process pdf file to python object
from PyPDF2 import PdfReader
def process_pdf(uploaded_pdf):
    pdf_file = PdfReader(uploaded_pdf)
    file_content = ""
    for page in pdf_file.pages:
        file_content += page.extract_text()

    return file_content

# Define a function that takes a document as an argument and return a Knowledge Graph
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

def doc2graph(prcoessed_pdf, llm):
    # Splitting texts to create chunks
    text_splitter = TokenTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 50,
    )

    text_chunks = text_splitter.split_text(prcoessed_pdf)
    # Creating a document object
    llm_transformer = LLMGraphTransformer(llm=llm)
    #graph_documents = llm_transformer.convert_to_graph_documents([Document(page_content=chunk) for chunk in text_chunks[:10]])
    graph_documents = llm_transformer.convert_to_graph_documents([Document(page_content=chunk) for chunk in text_chunks])

    return graph_documents

def add_nodes(graph_documents):
    with get_neo4j_connection().session(database=NEO4J_DATABASE) as session:
        kg.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        num_nodes = kg.query("MATCH (n) RETURN count(n)")
        print("Number of Nodes", num_nodes)

def get_neo4j_connection():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Define a function that clears the KG currently present in the AuraDB instance
def del_nodes():
  nodes_before_del = kg.query("MATCH (n) RETURN count(n)")
  del_cypher="MATCH (n) DETACH DELETE n"
  kg.query(del_cypher)
  count_nodes="MATCH (n) RETURN count(n)"
  return print("Number of nodes before", nodes_before_del, "Number of nodes after", kg.query(count_nodes))

# Loads graph to be displayed later
def load_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    session = driver.session()

    # Initialize the Pyvis Network
    net = Network(bgcolor="#222222", font_color="white")

    # Retrieve nodes and relationships
    default_cypher_nodes = "MATCH (n) RETURN n"
    default_cypher_rels = "MATCH (n)-[r]->(m) RETURN n, r, m"

    # Add nodes
    results_nodes = session.run(default_cypher_nodes)
    for record in results_nodes:
        node = record['n']
        node_id = node.element_id
        node_label = node.labels
        node_name = node.get('id', str(node_id))  # Adjust property access based on your data
        net.add_node(node_id, label=node_name, color="#FF5733")

    # Add edges
    results_rels = session.run(default_cypher_rels)
    for record in results_rels:
        start_node = record['n']
        end_node = record['m']
        relationship = record['r']
        start_node_id = start_node.element_id
        end_node_id = end_node.element_id
        rel_type = relationship.type
        net.add_edge(start_node_id, end_node_id, title=rel_type,  color="#33C1FF")

    # Save the graph to an HTML file
    graph_html_file = "graph.html"
    net.write_html(graph_html_file)

    # Close the Neo4j session
    session.close()

    return (graph_html_file)

# Show the loaded graph
def show_graph():
    knowledge_graph = load_graph()
    if knowledge_graph:
        with open(knowledge_graph, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600)
    else:
        st.error("Failed to load the graph.")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
## Convert the KG into Vecotr Embeddings
def vectorizeKG():
    vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    search_type = "hybrid",
    node_label = "Document",
    text_node_properties = ["text"],
    embedding_node_property="embedding",
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    )
    return vector_index

## Setup a chat prompt template for system prompt and user prompt
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

## I think this is where we define the entities that we want to track where our query should retrieve from

## Extract entities from text
class Entities(BaseModel):
  """Identifying information about Entities"""
  names: List[str] = Field(
      ...,
      description = """All the person, organization, or business entities that
      appear in the text""",
  )

## Prompt for extracting entities
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting oragniztion and person entities from the text",
        ),
        (
            "human",
            "Use the given format to extract information from the following"
            "input: {question}"
        ),
    ]
)

def create_entity_chain(llm):
    return prompt | llm.with_structured_output(Entities)

## Convert the user queries to cypher
match_query_nodes = '''MATCH (n)
RETURN id(n) AS id, labels(n) AS type'''

match_query_relationships = '''MATCH (n)-[r]->(m)
RETURN id(n) AS source_id, labels(n) AS source_type, id(m) AS target_id, labels(m) AS target_type, type(r) AS relationship_type'''

def map_to_database(entities: Entities) -> Optional[str]:
    result = ""
    
    # Query nodes
    node_response = kg.query(match_query_nodes)
    node_map = {record['id']: record['type'] for record in node_response}
    
    # Query relationships
    relationship_response = kg.query(match_query_relationships)
    for record in relationship_response:
        source_id = record['source_id']
        target_id = record['target_id']
        source_type = record['source_type']
        target_type = record['target_type']
        relationship_type = record['relationship_type']
        
        result += f"Relationship from {source_id} ({source_type}) to {target_id} ({target_type}) of type {relationship_type}\n"
    
    for entity in entities.names:
        node_type = node_map.get(entity, 'Unknown')
        result += f"{entity} is of type {node_type}\n"

    return result

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Generate Cypher statement based on natural language input
cypher_template = """Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:
- Use single quotes (') instead of backticks (`).
- Ensure the query is not empty.
- Do not include any Markdown formatting (e.g., no ```cypher blocks).
{schema}
Entities in the question map to the following database values:
{entities_list}
Question: {question}
Cypher query:"""  # noqa: E501

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        ("human", cypher_template),
    ]
)

# Truncate the schema to reduce the number of tokens
MAX_SCHEMA_LENGTH = 2000  # Adjust this value as needed
def get_truncated_schema():
    schema = kg.get_schema
    if len(schema) > MAX_SCHEMA_LENGTH:
        schema = schema[:MAX_SCHEMA_LENGTH] + "..."  # Indicate truncation
    return schema

def create_cypher_response_chain(llm):
    return (
        RunnablePassthrough.assign(names=create_entity_chain(llm))
        | RunnablePassthrough.assign(
            entities_list=lambda x: map_to_database(x["names"]),
            schema=lambda _: get_truncated_schema(),
        )
        | cypher_prompt
        | llm.bind(stop=["\nCypherResult:"])
        | StrOutputParser()
    )

## Create a chain using KG|cypher_prompt|LLM
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

# Cypher validation tool for relationship directions
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in kg.structured_schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)

# Generate natural language response based on database results
response_template = """Based on the the question, Cypher query, and Cypher response, write a natural language response:
Question: {question}
Cypher query: {query}
Cypher Response: {response}"""  # noqa: E501

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and Cypher response, convert it to a natural"
            " language answer. No pre-amble.",
        ),
        ("human", response_template),
    ]
)

def create_main_chain(llm):
    return (
        RunnablePassthrough.assign(query=create_cypher_response_chain(llm))
        | RunnablePassthrough.assign(
            response=lambda x: kg.query(cypher_validation(x["query"])),
        )
        | response_prompt
        | llm
        | StrOutputParser()
    )

def main():
    st.title("Graph RAG")

    # Initialize session state for storing graph data and user input
    if "graph_html_file" not in st.session_state:
        st.session_state.graph_html_file = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "llm" not in st.session_state:
        st.session_state.llm = None

    # Upload File
    uploaded_file = st.file_uploader("Upload file to convert into Knowledge Graph", type=['pdf'], key="file_uploader")

    # Handle file upload
    if uploaded_file is not None:
        # Toggle switch for choosing between OpenAI and Groq
        use_openai = st.toggle("Use OpenAI (toggle off for Groq)", value=True)

        proceed = False
        user_api_key = None

        if use_openai:
            user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if user_api_key:
                proceed = True
            else:
                st.warning("Please enter your OpenAI API key to proceed.")
        else:
            if groq_api_key:
                proceed = True
            else:
                st.error("Groq API key is not set in the environment variables.")

        if proceed:
            # Initialize LLM
            if st.session_state.llm is None:
                st.session_state.llm = initialize_llm(user_api_key, use_openai)

            # Show the "Show Graph" button
            if not st.session_state.file_processed:
                if st.button('Show Graph'):
                    with st.spinner('Loading graph...'):
                        del_nodes()
                        processed_pdf = process_pdf(uploaded_file)
                        graph_doc = doc2graph(processed_pdf, st.session_state.llm)
                        add_nodes(graph_doc)  # This now includes retry logic
                        st.session_state.graph_html_file = load_graph()
                        st.session_state.file_processed = True

    # Display the graph if it's available in session state
    if st.session_state.graph_html_file and st.session_state.llm:
        st.title("Knowledge Graph")
        # Load and display the graph from the stored HTML file
        with open(st.session_state.graph_html_file, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600)

        # Vector search from the KG
        st.title('Retrieve from Graph')
        # Convert the KG to Vector Index
        vectorizeKG()

        # Display a text input field and get the user's input
        st.session_state.user_input = st.text_input("Enter your text here:", st.session_state.user_input)

        # Display the input after the user submits it
        if st.session_state.user_input:
            # Create the chain using the LLM from session state
            chain = create_main_chain(st.session_state.llm)
            response = chain.invoke({"question": st.session_state.user_input})
            st.write(response)

if __name__ == "__main__":
    main()

