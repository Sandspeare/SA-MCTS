from neo4j import GraphDatabase

# Modify it with your database connection and authentication information
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))

def import_entities(tx, scene):
    # Construct the path to the CSV file of entity types for the scene (note that the file name should match the actual one)
    query = f"""
    LOAD CSV WITH HEADERS FROM 'file:///场景{scene}的实体类型.csv' AS row
    MERGE (n:Entity {{name: row.subject}})
    SET n.scene = '{scene}', 
        n.relationInfo = row.relation, 
        n.extra = row.object
    """
    tx.run(query)

def import_relations(tx, scene):
    # Construct the path to the CSV file of relation types for the scene
    query = f"""
    LOAD CSV WITH HEADERS FROM 'file:///场景{scene}的关系类型.csv' AS row
    MATCH (a:Entity {{name: row.subject}}), (b:Entity {{name: row.object}})
    MERGE (a)-[r:RELATION {{scene: '{scene}', type: row.relation}}]->(b)
    """
    tx.run(query)

with driver.session() as session:
    # Iterate through scenes 1 to 35
    for scene in range(1, 36):
        print(f"Importing entity data for scene {scene}...")
        session.write_transaction(import_entities, scene)
        print(f"Importing relation data for scene {scene}...")
        session.write_transaction(import_relations, scene)
    print("All data has been imported!")

driver.close()