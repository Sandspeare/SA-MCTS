from StoryPathGenerator import Neo4jConnector, StoryPathGenerator
from SuspenseEvaluator import SuspenseScorer
from NarrativeIntegrator import NarrativeIntegrator
import json

def generate_story():
    # Connect to Neo4j
    neo4j_conn = Neo4jConnector("bolt://localhost:7687", "neo4j", "12345678")

    # Initialize the generator
    generator = StoryPathGenerator(
        neo4j_conn=neo4j_conn,
        initial_scene_id="CrimeScene",
        max_depth=12
    )

    # Generate the plot line of the story occurrence and the detective investigation line respectively
    plot_line = generator.generate_plot_line(simulations=1500)
    detective_line = generator.generate_detective_line(simulations=1500)

    # Integrate the two storylines and inject the reasoning line
    integrator = NarrativeIntegrator(neo4j_conn)
    integrator.integrate_storylines(plot_line, detective_line)

    # Assume that at this time we determine the final best path, and you can design your own integration logic to select the best path here
    # For example, select the final path best_path after comprehensively evaluating the scores of the two lines
    # Here, simply take the path of the story occurrence line as an example
    best_path = [node.scene['id'] for node in plot_line]
    print(f"\nBest Path (with the highest suspense score):\n{' -> '.join(best_path)}")

    # Construct the LLM prompt for generating the novel
    # Note that this includes all the key information of the two integrated routes and the reasoning line
    prompt = f"""
story line: {' -> '.join([node.scene['id'] for node in plot_line])}
Detective Line: {' -> '.join([node.scene['id'] for node in detective_line])}
Lines of reasoning: cab driver + poison + love interest + RACHE + tall, rosy-cheeked, brown coat ==> murderer.
Ultimate goal: catch the murderer
Please generate a complete narrative text of a mystery novel based on the above information.
    """

    # Call the LLM evaluator to generate the novel (assuming the _validate_with_llm method accepts the prompt or path information)
    narrative_text = generator.generate_dual_story_text(
        plot_line=[n for n in plot_line if n.scene['type'] == 'plot'],
        detective_line=[n for n in detective_line if n.scene['type'] == 'detective'],
        reasoning_clues=["taxi driver", "poisoning", "rival in love", "RACHE", "tall, ruddy-complexioned, brown coat"]
    )

    print("\nGenerated novel content:")
    # If narrative_text is a dict, convert it to a JSON string and then output it (otherwise, output it directly)
    if isinstance(narrative_text, dict):
        narrative_text = json.dumps(narrative_text, ensure_ascii=False, indent=2)
    print(narrative_text)

    with open("generated_story.txt", "w", encoding="utf-8") as f:
        f.write(narrative_text)

    # Optional: Generate a DOT file for path visualization
    dot_graph = generator.visualize_path(best_path)
    with open("best_path.dot", "w", encoding="utf-8") as f:
        f.write(dot_graph)

if __name__ == "__main__":
    generate_story()