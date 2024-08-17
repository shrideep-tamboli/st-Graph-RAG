from pyvis.network import Network
from neo4j import GraphDatabase
import streamlit as st
import os
from dotenv import load_dotenv  

load_dotenv()
# Setup connection to Neo4j database
from langchain_community.graphs import Neo4jGraph
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
groq_api_key = os.getenv("groq_api_key")


# Setup llm
from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")


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
def doc2graph(prcoessed_pdf):
    # Splitting texts to create chunks
    text_splitter = TokenTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 50,
    )

    text_chunks = text_splitter.split_text(prcoessed_pdf)
    # Creating a document object
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents([Document(page_content=chunk) for chunk in text_chunks[:5]])

    return graph_documents


# Define a function to add the converted KG to AuraDB instance
def add_nodes(graph_documents):
    kg.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True)    
    num_nodes = kg.query("MATCH (n) RETURN count(n)")
    return print("Number of Nodes", num_nodes)


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
    net = Network()

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
        net.add_node(node_id, label=node_name)

    # Add edges
    results_rels = session.run(default_cypher_rels)
    for record in results_rels:
        start_node = record['n']
        end_node = record['m']
        relationship = record['r']
        start_node_id = start_node.element_id
        end_node_id = end_node.element_id
        rel_type = relationship.type
        net.add_edge(start_node_id, end_node_id, title=rel_type)

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

def main():
    st.title("Graph RAG")

    ## Upload File
    uploaded_file = st.file_uploader("Upload file to convert into Knowledge Graph", type=['pdf'])
    
    ## Process the Uploaded_file to a python object


    ## If a file has been uploaded then:    
    if uploaded_file is not None:
        ## As soon as file is uploaded 
        # 1. Call function to delete KG from AuraDB
        del_nodes()

        # 2. Display a button to load the KG in AuraDB
        ## On_click call function to convert the Doc to KG and Add it to the AuraDB instance
        if st.button('Show Graph'):
            # Show loading spinner while processing and displaying the graph
            with st.spinner('Loading graph...'):
                # Process the uploaded PDF to python object
                processed_pdf = process_pdf(uploaded_file)
                # Convert the processed pdf to Knowledge Graph
                graph_doc = doc2graph(processed_pdf)
                # Add the graph document
                add_nodes(graph_doc)
                # Displat the KG
                show_graph()

if __name__ == "__main__":
    main()
