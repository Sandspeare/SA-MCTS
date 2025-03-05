# Suspense Story Generation System

    This project aims to generate suspense-style novel story  by integrating **Situation Graph**, **Monte Carlo Tree Search (MCTS)**, **Large Language Models (LLMs)**, and various scene evaluation and optimization techniques. The core functions of the project include:


- **Story Path Generation**：Utilize the Neo4j graph database to construct scene relationships and search for the optimal story path based on        MCTS.
- **Suspense Evaluation**：Calculate and score suspense indicators for scenes, comprehensively considering uncertainty, threat level, escape limitations, and narrative techniques.
- **LLMs Verification**： Evaluate the generated path descriptions by invoking the LLM and obtain improvement suggestions.
- **Path Optimization and Integration**：Clean up redundancies, enhance the rhythm, and insert necessary elements for the original story path. Also, support the integration of the main storyline and the detective storyline to construct a complete story.

---

## Project Structure
├── config.py # Global configuration file, including API Key, number of retries, MCTS configuration, relationship penalties, and safety thresholds, etc.
├── llm_evaluator.py # LLM evaluator, constructs prompts to call the OpenAI interface and parses the returned JSON results.
├── Node.py # Defines a unified SuspenseNode class and SuspenseMetrics.
├── SuspenseEvaluator.py # Suspense scorer, calculates the comprehensive suspense score of the path based on node metrics (including time, narrative, and penalty factors).
├── StoryPathGenerator.py # Story path generator: Utilizes the Neo4j connector, MCTS search, and LLM evaluation to generate story paths, and also supports path visualization.
├── PathOptimizer.py # Path optimizer, cleans up redundant scenes, enhances the rhythm, and inserts missing necessary elements to ensure that the generated path meets the expected constraints.
├── NarrativeIntegrator.py # Integrates different storylines (such as the main storyline and the detective investigation storyline) to generate the final intertwined story path.
├── MCTSController.py # Enhanced Monte Carlo Tree Search (MCTS) controller, responsible for path search, simulation, and backpropagation, and provides the best path.
├── test_node.py # Suspense node test example, verifies basic indicator calculations and relationship penalties (such as causal relationship scoring).
└── test_story_path.py # Story path generation test example, tests the Neo4j database connection and the integrity of path generation.
---

## Installation and Configuration

1. **Dependency Installation**  
   Please ensure that the following Python libraries are installed (or specified in requirements.txt):
   - neo4j
   - openai
   - pytest
   - Other required libraries (e.g., math, random, dataclasses, etc., which are all Python standard libraries)

2. **Configuration File**  
   Set your OpenAI API Key and Neo4j database connection parameters and other configuration information in  `config.py` . 
   > It is recommended to use environment variables to hide sensitive information.

3. **Database Preparation**  
   Please ensure that the Neo4j database has been set up and contains the necessary  `Scene` nodes and relationships. The test files ( such as `test_story_path.py`）have examples of cleaning and constructing test data.

---

## Usage

- **Generate Story Path**  
  Call the `StoryPathGenerator` class in the main program:
  ```python
  from StoryPathGenerator import Neo4jConnector, StoryPathGenerator
  
  connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "your_password")
  generator = StoryPathGenerator(connector, "start_scene")
  # Generate a story path (which can be verified by the LLM or directly return a list of scene IDs)
  story_path = generator.generate_story_path(simulations=1000)
  print("The generated story path：", story_path)
  connector.close()

Plot Optimization and Integration
Optimize the generated path through PathOptimizer, and integrate the main storyline and the detective storyline through NarrativeIntegrator to generate the final integrated story path.
Suspense Scoring
Use SuspenseEvaluator to evaluate the overall suspense level of the generated scene path, and provide various indicators and comprehensive scores.
Testing
Use pytest to run test_node.py and test_story_path.py respectively to verify the functions of each module.