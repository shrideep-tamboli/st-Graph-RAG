from neo4j import GraphDatabase

NEO4J_URI = "neo4j+ssc://82593fd3.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "igPFpSEjY82ZDUDAFc2joQ2TF-_NpzF4DWopg7EmhAw"

# Create the driver with SSL enabled by the URI scheme
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Start the session and run a test query
session = driver.session()

try:
    result = session.run("RETURN 1")
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    session.close()
