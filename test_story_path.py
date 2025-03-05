# test_story_path.py
import pytest

from StoryPathGenerator import Neo4jConnector, StoryPathGenerator


def test_basic_path_generation():
    # Set up test data
    connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "12345678")

    try:
        # Clean up existing data
        connector.execute_query("MATCH (n) DETACH DELETE n")

        # Create test scenes and relationships
        setup_queries = [
            """
            CREATE (start:Scene {id: 'start_scene'})
            CREATE (middle:Scene {id: 'middle_scene'})
            CREATE (end:Scene {id: 'end_scene'})
            CREATE (start)-[:NEXT_SCENE {weight: 1.0}]->(middle)
            CREATE (middle)-[:NEXT_SCENE {weight: 1.0}]->(end)
            """
        ]

        for query in setup_queries:
            connector.execute_query(query)

        # Execute the test
        generator = StoryPathGenerator(connector, "start_scene")
        path = generator.generate_story_path(simulations=1000)

        # Clean up test data
        connector.execute_query("MATCH (n) DETACH DELETE n")s

        assert len(path) >= 3, "Generate at least the basic path"
        assert path[0] == "start_scene", "The starting scene is correct"

    finally:
        #  Ensure the connection is closed
        connector.close()
