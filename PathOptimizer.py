import random
from typing import List, Optional
from Node import SuspenseNode


class PathOptimizer:
    def __init__(self, relations_graph, max_depth=15):
        self.graph = relations_graph  # Story relationship graph
        self.max_depth = max_depth

        # Suspense contribution of different scene types
        self.type_weights = {
            'clue': 0.8,
            'action': 0.6,
            'dialogue': 0.4,
            'reflection': 0.3,
            'twist': 1.0
        }

        # Twist configuration
        self.twist_config = {
            'min_intensity_diff': 0.1,  # # Minimum intensity difference
            'max_consecutive_similar': 2  # Maximum number of consecutive similar scenes
        }

    def optimize_path(self, path: List[SuspenseNode]) -> List[SuspenseNode]:
        """Execute a three - step optimization process"""
        optimized_path = self._clean_redundancies(path)
        optimized_path = self._enhance_pacing(optimized_path)
        optimized_path = self._insert_missing_elements(optimized_path)
        return optimized_path[:self.max_depth]  # Ensure it does not exceed the maximum depth

    def _clean_redundancies(self, path: List[SuspenseNode]) -> List[SuspenseNode]:
        """Clean up redundant scenes"""
        unique_scenes = []
        seen = set()
        similar_count = 0  # Track consecutive similar scenes

        for node in path:
            scene_id = node.scene['id']
            scene_type = node.scene.get('type')

            # Check for ID duplication
            if scene_id in seen:
                alternative = self._find_alternative(node, unique_scenes)
                if alternative:
                    unique_scenes.append(alternative)
                    similar_count = 0
                continue

            # Check for consecutive similar scenes
            if unique_scenes and scene_type == unique_scenes[-1].scene.get('type'):
                similar_count += 1
                if similar_count >= self.twist_config['max_consecutive_similar']:
                    alternative = self._find_alternative(node, unique_scenes)
                    if alternative:
                        unique_scenes.append(alternative)
                        similar_count = 0
                    continue
            else:
                similar_count = 0

            unique_scenes.append(node)
            seen.add(scene_id)

        return unique_scenes

    def _find_alternative(self, node: SuspenseNode,
                          current_path: List[SuspenseNode]) -> Optional[SuspenseNode]:
        """寻Find an alternative scene"""
        # Get related scenes from the relationship graph
        related_scenes = self.graph.get_scene_relations(node.scene['id'])

        # Filter out used scenes
        used_ids = {n.scene['id'] for n in current_path}
        candidates = [
            rel for rel in related_scenes
            if rel['target'] not in used_ids
               and rel['weight'] >= 0.5  # Ensure the relationship strength is sufficient
        ]

        if not candidates:
            return None

        # Sort candidates by relationship weight and select
        sorted_candidates = sorted(candidates, key=lambda x: x['weight'], reverse=True)
        selected = sorted_candidates[0]

        # Construct a new node
        new_scene = self.graph.get_scene_by_id(selected['target'])
        return SuspenseNode(new_scene)

    def _enhance_pacing(self, path: List[SuspenseNode]) -> List[SuspenseNode]:
        """Enhance pacing control"""
        if not path:
            return path

        new_path = []
        intensity_map = []  # Record the suspense intensity change curve

        for i, node in enumerate(path):
            # Calculate the current scene intensity
            intensity = (node.uncertainty + node.threat_level) * 0.5
            intensity_map.append(intensity)

            # Check if a twist is needed
            if i > 0 and self._needs_twist(intensity_map):
                twist_node = self._create_twist_node(node, intensity_map)
                if twist_node:
                    new_path.append(twist_node)

            new_path.append(node)

        return new_path

    def _needs_twist(self, intensity_map: List[float]) -> bool:
        """Determine if a twist is needed"""
        if len(intensity_map) < 2:
            return False

        # Check the intensity change
        recent_change = abs(intensity_map[-1] - intensity_map[-2])
        return recent_change < self.twist_config['min_intensity_diff']

    def _create_twist_node(self, current_node: SuspenseNode,
                           intensity_map: List[float]) -> Optional[SuspenseNode]:
        """Create a twist scene"""
        current_intensity = intensity_map[-1]

        # Get possible twist scenes
        twist_candidates = self.graph.query_scenes(
            type='twist',
            min_intensity=current_intensity + 0.2  # 确保强度提升
        )

        if not twist_candidates:
            return None

        # Select the most suitable twist
        selected = max(twist_candidates,
                       key=lambda x: x.get('dramatic_impact', 0))

        return SuspenseNode(selected)

    def _insert_missing_elements(self, path: List[SuspenseNode]) -> List[SuspenseNode]:
        """Insert missing mandatory elements"""
        mandatory_elements = {
            'evidence_discovery': lambda p: any(n.scene['type'] == 'clue' for n in p),
            'investigation_action': lambda p: sum(1 for n in p if n.scene['type'] == 'action') >= 2
        }

        result_path = path.copy()

        for element, check in mandatory_elements.items():
            if not check(result_path):
                #  Get candidate scenes
                candidate = self.graph.get_rand_scene(element)
                if candidate:
                    # Find the best insertion position
                    insert_pos = self._find_best_insert_pos(result_path, candidate)
                    result_path.insert(insert_pos, SuspenseNode(candidate))

        return result_path

    def _find_best_insert_pos(self, path: List[SuspenseNode],
                              candidate: dict) -> int:
        """Find the best insertion position"""
        if not path:
            return 0

        best_pos = 0
        best_score = float('-inf')

        # Try each possible position
        for i in range(len(path) + 1):
            test_path = path.copy()
            test_path.insert(i, SuspenseNode(candidate))
            score = self._evaluate_insertion(test_path, i)

            if score > best_score:
                best_score = score
                best_pos = i

        return best_pos

    def _evaluate_insertion(self, path: List[SuspenseNode], pos: int) -> float:
        """Evaluate the rationality of the insertion position"""
        if pos == 0 or pos == len(path):
            return 0.0

        prev_node = path[pos - 1]
        inserted_node = path[pos]
        next_node = path[pos + 1]

        # Evaluate coherence
        coherence = self._evaluate_coherence(prev_node, inserted_node, next_node)

        # Evaluate pacing
        pacing = self._evaluate_pacing(path, pos)

        # Evaluate type distribution
        type_distribution = self._evaluate_type_distribution(path)

        return coherence * 0.4 + pacing * 0.4 + type_distribution * 0.2

    def _evaluate_coherence(self, prev: SuspenseNode,
                            current: SuspenseNode,
                            next: SuspenseNode) -> float:
        """Evaluate scene coherence"""
        # Implement specific coherence evaluation logic
        return 0.5  # Example return value

    def _evaluate_pacing(self, path: List[SuspenseNode], pos: int) -> float:
        """Evaluate pacing control"""
        # Implement specific pacing evaluation logic
        return 0.5  # Example return value

    def _evaluate_type_distribution(self, path: List[SuspenseNode]) -> float:
        """Evaluate scene type distribution"""
        # Implement specific type distribution evaluation logic
        return 0.5  #  Example return value
        