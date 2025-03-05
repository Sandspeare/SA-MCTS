from neo4j import GraphDatabase
import math
import random
from collections import defaultdict
from config import config
from llm_evaluator import LLMEvaluator
from typing import List

from MCTSController import EnhancedMCTS
from Node import SuspenseNode

class Neo4jConnector:
    """Neo4j Knowledge Graph Connector"""
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_query(self, query, parameters=None):
        """
        Execute a Neo4j query
        
        Args:
            query (str): Cypher query statement
            parameters (dict, optional): Query parameters
            
        Returns:
            list: List of query results
        """
        if parameters is None:
            parameters = {}
            
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return list(result)

    def get_scene_relations(self, scene_id):
        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run(
                    "MATCH (s:Scene {name:$scene_id})-[r]->(t:Scene) "
                    "RETURN t.name as target, r.type as rel_type, r.note as note, r.weight as weight",
                    scene_id=scene_id
                ).data()
            )
        return result

    def get_next_scenes(self, scene_id):
        """
        Get all possible next scenes for a given scene
        
        Args:
            scene_id (str): Current scene ID
            
        Returns:
            list: List of next scenes, each scene contains an 'id' attribute
        """
        query = """
        MATCH (current:Scene {id: $scene_id})-[:NEXT_SCENE]->(next:Scene)
        RETURN next.id as id
        """
        with self.driver.session() as session:
            result = session.run(query, scene_id=scene_id)
            return [record for record in result]

    def close(self):
        """Close the database connection"""
        if self.driver is not None:
            self.driver.close()

class KnowledgeGraphNode:
    """Enhanced Graph Node Representation"""
    RELATION_WEIGHTS = {
        ('CAUSES', 'explicit'): 1.0,
        ('CAUSES', 'implicit'): 0.7,
        ('TIME_SEQUENCE', 'main'): 0.6,
        ('TIME_SEQUENCE', 'branch'): 0.5
    }

    def __init__(self, scene_id, neo4j_conn):
        self.scene_id = scene_id
        self.conn = neo4j_conn
        self.relations = self._load_relations()

    def _load_relations(self):
        raw_relations = self.conn.get_scene_relations(self.scene_id)
        processed = []
        for rel in raw_relations:
            # Add null value check
            if rel is None or 'rel_type' not in rel or rel['rel_type'] is None:
                continue
            relation_type = rel['rel_type'].split('_')[0]
            detail_type = '_'.join(rel['rel_type'].split('_')[1:])
            base_weight = self.RELATION_WEIGHTS.get((relation_type, detail_type), 0.5)
            priority = 1 if 'Key Causality' in rel['note'] else 2 if 'Secondary Causality' in rel['note'] else 3
            final_weight = base_weight * (1 - 0.2 * (priority - 1))
            processed.append({
                'target': rel['target'],
                'type': relation_type,
                'detail': detail_type,
                'weight': final_weight,
                'priority': priority,
                'note': rel['note']
            })
        return processed

    def get_next_scene(self):
        """
         Get the next scene node
        
        Returns:
            KnowledgeGraphNode: The next scene node, or None if there is no next scene
        """
        if not self.relations:
            return None
            
        # Get all possible next scenes
        next_scenes = self.conn.get_next_scenes(self.scene_id)
        if not next_scenes:
            return None
            
        # Randomly select a next scene
        next_scene = random.choice(next_scenes)
        return KnowledgeGraphNode(next_scene['id'], self.conn)




class StoryPathGenerator:
    def __init__(self, neo4j_conn, initial_scene_id, max_depth=15):
        self.neo4j_conn = neo4j_conn
        self.max_depth = max_depth
        self.initial_scene = self._build_initial_node(initial_scene_id)
        self.narrative_patterns = defaultdict(float)

        # Modify the weight structure to ensure each relation type has subtypes
        self.relation_weights = {
            'CAUSES': {
                'explicit': 1.0,
                'implicit': 0.7
            },
            'TIME_SEQUENCE': {
                'main': 0.6,
                'branch': 0.5
            },
            'OTHER': {
                'default': 0.4
            }
        }
        self.priority_weights = {1: 1.0, 2: 0.8, 3: 0.6}
        self.mcts = self._initialize_mcts()
        # Initialize the LLM evaluator
        self.llm_evaluator = LLMEvaluator()

    def _build_initial_node(self, scene_id):
        kg_node = KnowledgeGraphNode(scene_id, self.neo4j_conn)
        # Construct the initial scene, add necessary fields to ensure normal initialization of SuspenseNode
        scene = {
            'id': scene_id,
            'relations': kg_node.relations,
            'priority': 1,
            'type': self._detect_scene_type(kg_node),
            'children': [rel for rel in kg_node.relations],  # Assume each relationship corresponds to a candidate sub - scene
            # Other scene attributes
            'uncertainty': 0.5,
            'threat_level': 0.3,
            'escape_restriction': 0.4,
            'narrative_techniques': [],
            'duration': 30
        }
        return SuspenseNode(scene)

    def _generate_clue_chain(self, path_nodes):
        """Build a logical clue chain from scattered scenes""""
        clue_map = {
            'taxi': [],
            'poison': [],
            'rache': []
        }
        for node in path_nodes:
            for clue in node.scene.get('clues', []):
                clue_lower = clue.lower()
                if clue_lower in clue_map:
                    clue_map[clue_lower].append(node.scene.get('id'))
        prompt = "Even if there is no direct relationship between scenes, please infer the murderer through the following clue chains:\n"
        prompt += f"Taxi traces appear in scenes{clue_map['taxi']}\n"
        prompt += f"Poison detection is triggered in scenes {clue_map['poison']}\n"
        prompt += f"The blood - word RACHE appears in scenes  {clue_map['rache']}\n"
        return prompt

    def _calculate_relation_weight(self, relation_type: str, relation_detail: str = None) -> float:
        """Calculate the relation weight"""
        if relation_type not in self.relation_weights:
            return self.relation_weights['OTHER']['default']
        relation_subtypes = self.relation_weights[relation_type]
        if relation_detail in relation_subtypes:
            return relation_subtypes[relation_detail]
        return relation_subtypes.get('default', 0.4)

    def generate_story_path(self, simulations=1000):
        """
        Generate a story path using MCTS or LLM according to the configuration flag.
        """
        """
        if config.ENABLE_LLM_VALIDATION:
            optimized_path = self.mcts.get_best_path(simulations)
            llm_prompt = self._build_prompt(optimized_path)
            return self._validate_with_llm(llm_prompt)
        else:
            return [node.scene['id'] for node in self.mcts.get_best_path(simulations)]

    def _validate_with_llm(self, prompt: str):
        """Use the LLM evaluator to evaluate the path description and return the LLM's processing result"""
        return self.llm_evaluator.evaluate_path(prompt)

    def visualize_path(self, path):
        """生Generate a DOT - format string for path visualization"""
        dot_content = ["digraph G {", "    rankdir=LR;"]
        for i, scene_id in enumerate(path):
            node = self._get_node_by_id(scene_id)
            label = f"{scene_id}\\n{node.scene.get('type', 'Unknown')}"
            dot_content.append(f'    node{i} [label="{label}"];')
        for i in range(len(path) - 1):
            dot_content.append(f"    node{i} -> node{i + 1};")
        dot_content.append("}")
        return "\n".join(dot_content)

    def _detect_scene_type(self, kg_node):
        causes_count = sum(1 for r in kg_node.relations if r['type'] == 'CAUSES')
        time_seq_count = sum(1 for r in kg_node.relations if r['type'] == 'TIME_SEQUENCE')
        if causes_count >= 2:
            return 'DECISION_POINT'
        elif time_seq_count > 0 and causes_count > 0:
            return 'HYBRID_NODE'
        return 'DEFAULT'

    def _build_prompt(self, path_nodes):
        """Construct an LLM prompt based on the path nodes"""
        prompt_lines = ["Generate story narrative with the following constraints:"]
        for node in path_nodes:
            # Assume the node has relation_types and metrics (including uncertainty)
            prompt_lines.append(
                f"- Scene {node.scene['id']}: Relations=[{', '.join(getattr(node, 'relation_types', []))}], "
                f"SuspenseScore={getattr(node, 'metrics', {}).get('uncertainty', 0.0):.2f}"
            )
        # Example: Map some IDs to clue keywords
        clue_markers = {2: "taxi", 31: "poison", 32: "RACHE"}
        clues_included = [clue_markers.get(n.scene['id'], "") for n in path_nodes]
        clues_included = list(filter(None, clues_included))
        prompt_lines.append("\nMust include these clues: " + ", ".join(clues_included))
        return "\n".join(prompt_lines)

    def generate_dual_story_text(self, plot_line, detective_line, reasoning_clues):
        """Generate novel text based on the dual - line structure and reasoning clues"""
        # Build a dual - line LLM prompt
        prompt = f"""
    【Story Plot Line】: {' → '.join([n.scene['id'] for n in plot_line])}
    【Detective Investigation Line】: {' → '.join([n.scene['id'] for n in detective_line])}
    【Reasoning Line Features】: {' + '.join(reasoning_clues)} => The Murderer 

    Generate requirements:
    1. each section is divided by @
    2. two lines of progress need to be cross-presented
    3. the end must be wrapped in the <final reasoning> tag to characterize the convergence process.
    4. Please generate a detailed novel narrative text of no less than 3,000 words “each chapter is full of specific content and details.
    5. The text of the output can be in English.
        """
        # Bypass validation and directly call the native generation <source_id>StoryPathGenerator.py</source_id>
        return self.llm_evaluator.direct_generate(prompt)

    def _extract_clues_from_path(self, scenes: List['SuspenseNode']) -> str:
        clues = []
        for scene in scenes:
            scene_clues = scene.scene.get('clues', [])
            if scene_clues:
                clues.extend(scene_clues)
        return ', '.join(clues) if clues else "No clues available"

    def get_next_scene(self, current_node):
        """
      Get the next scene node
        """
        next_scenes = self.neo4j_conn.get_next_scenes(current_node.scene.get('id'))
        if not next_scenes:
            return None
        next_scene = random.choice(next_scenes)
        return SuspenseNode({'id': next_scene['id']}, parent=current_node)

    def _initialize_mcts(self):
        return EnhancedMCTS(
            root=self.initial_scene,
            relation_weights=self.relation_weights,
            priority_weights=self.priority_weights,
            max_depth=self.max_depth
        )

    def _expand_node(self, node):
        possible_relations = node.scene.get('relations', [])
        filtered = [r for r in possible_relations if self._check_coherence(r, node)]
        sorted_rels = sorted(filtered, key=lambda r: (
            r['priority'],
            self.relation_weights.get(r['type'], {}).get(r.get('detail', ''), 0.5),
            random.random()
        ), reverse=True)
        if sorted_rels:
            selected = sorted_rels[0]
            new_scene = self._create_scene_from_relation(selected)
            new_node = SuspenseNode(new_scene, parent=node)
            node.children.append(new_node)
            return new_node
        return node

    def _check_coherence(self, relation, parent_node):
        if parent_node.parent and parent_node.parent.scene.get('type') == relation.get('type'):
            return False
        if any(child.scene.get('id') == relation['target'] for child in parent_node.children):
            return False
        return True

    def _create_scene_from_relation(self, relation):
        kg_node = KnowledgeGraphNode(relation['target'], self.neo4j_conn)
        scene = {
            'id': relation['target'],
            'relations': kg_node.relations,
            'priority': relation['priority'],
            'type': self._detect_scene_type(kg_node),
            'children': [rel for rel in kg_node.relations]
        }
        # If there is a key hint in the relation, mark it as an inference line (with preset clues)
        if relation.get('note') and "关键线索" in relation['note']:
            scene['is_inference'] = True
            scene['clues'] = ['Taxi Driver', 'Poisoning', 'Love Rival', 'RACHE', 'Tall', 'Ruddy Complexion', 'Brown Coat']
        return scene

    def _adjust_weights_based_on_pattern(self, path):
        if not path:
            return
        pattern_components = []
        for node in path:
            rel_counts = defaultdict(int)
            for rel in node.scene.get('relations', []):
                rel_type = rel.get('type', 'OTHER')
                rel_detail = rel.get('detail', 'default')
                key = f"{rel_type}_{rel_detail}"
                rel_counts[key] += 1
            if rel_counts:
                pattern_components.append(max(rel_counts, key=rel_counts.get))
        for relation_type in self.relation_weights:
            occurrence = sum(1 for pc in pattern_components if pc.startswith(relation_type))
            adjust_factor = 1 + math.log(1 + occurrence) / self.max_depth
            for subtype in self.relation_weights[relation_type]:
                current_weight = self.relation_weights[relation_type][subtype]
                self.relation_weights[relation_type][subtype] = current_weight * adjust_factor

    def _extract_best_path(self, root):
        path = []
        current = root
        depth = 0
        while current and depth < self.max_depth:
            path.append(current)
            if not current.children:
                break
            current = max(current.children, key=lambda c: c.score / c.visits if c.visits > 0 else 0)
            depth += 1
        seen = set()
        final_path = []
        for node in path:
            if node.scene.get('id') in seen:
                break
            seen.add(node.scene.get('id'))
            final_path.append(node)
        return final_path

    def _postprocess_path(self, raw_path):
        processed = []
        for node in raw_path:
            if any(n.scene.get('id') == node.scene.get('id') for n in processed):
                continue
            processed.append(node)
        return [n.scene.get('id') for n in processed]

    def _simulate(self, node):
        return random.uniform(0, 1)

    def _backpropagate(self, node, reward):
        current = node
        while current is not None:
            current.visits += 1
            current.score += reward
            current = current.parent

    def _get_node_by_id(self, scene_id):
        query = """
        MATCH (s:Scene {id: $scene_id})
        RETURN s
        """
        result = self.neo4j_conn.execute_query(query, {'scene_id': scene_id})
        if not result:
            scene = {
                'id': scene_id,
                'type': 'unknown',
                'uncertainty': 0.5,
                'threat_level': 0.3,
                'escape_restriction': 0.4,
                'narrative_techniques': [],
                'duration': 30
            }
        else:
            scene_data = result[0]['s']
            scene = {
                'id': scene_id,
                'type': scene_data.get('type', 'unknown'),
                'uncertainty': scene_data.get('uncertainty', 0.5),
                'threat_level': scene_data.get('threat_level', 0.3),
                'escape_restriction': scene_data.get('escape_restriction', 0.4),
                'narrative_techniques': scene_data.get('narrative_techniques', []),
                'duration': scene_data.get('duration', 30)
            }
        return SuspenseNode(scene)


    def generate_plot_line(self, simulations=1500):
        """
        Generate the story plot line. Directly use the default MCTS search strategy to generate the path.
        """
        best_path = self.mcts.get_best_path(simulations)
        return best_path

    def generate_detective_line(self, simulations=1500):
        """
        Generate the detective investigation line. Strategy:
         1. Conduct multiple simulations to generate candidate paths (e.g., 5 times).
         2. Count the number of scenes containing detective-related elements in each candidate path:
            Here, it is judged based on whether there is a "clues" list in the scene.
         3. Select and return the candidate path with the highest proportion of detective scenes.
        """
        iterations = 5  # The number of times to simulate candidate paths
        best_ratio = -1.0
        best_candidate = None

        for i in range(iterations):
            candidate_path = self.mcts.get_best_path(simulations=simulations // iterations)
            if not candidate_path or len(candidate_path) == 0:
                continue
            # Count the number of nodes containing "clues", which are regarded as detective-related scenes
            detective_count = sum(1 for node in candidate_path if node.scene.get("clues"))
            ratio = detective_count / len(candidate_path)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate_path

        if best_candidate is None:
            best_candidate = self.mcts.get_best_path(simulations)
        return best_candidate