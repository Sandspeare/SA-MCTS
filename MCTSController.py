# MCTSController.py
import math
import random
from collections import defaultdict
from typing import List, Tuple, Optional
from Node import SuspenseNode

class EnhancedMCTS:
    def __init__(self, root: SuspenseNode, exploration_weight=1.414, 
                 relation_weights=None, priority_weights=None, max_depth=15):
        self.root = root
        self.exploration_weight = exploration_weight
        self.relation_weights = relation_weights or {}
        self.priority_weights = priority_weights or {}
        self.max_depth = max_depth
        self.narrative_memory = defaultdict(float)

    def get_best_path(self, simulations=1000):
        for _ in range(simulations):
            node, path = self.select()
            if not node.is_terminal():
                node = self.expand(node)
                simulation_score = self.simulate(node)
                self.backpropagate(node, simulation_score, path)
        return self._extract_best_path()

    def select(self):
        path = []
        current = self.root
        while not current.is_terminal() and current.is_fully_expanded():
            current = self._best_child(current)
            path.append(current)
            if len(path) > self.max_depth:
                break
        return current, path

    def _best_child(self, node):
        best_score = float('-inf')
        best_children = []
        for child in node.children:
            score = self._calculate_uct_score(child, node)
            if score > best_score:
                best_score = score
                best_children = [child]
            elif abs(score - best_score) < 1e-6:
                best_children.append(child)
        return random.choice(best_children)

    def _calculate_uct_score(self, child, parent):
        relation_weight = self.relation_weights.get(
            child.scene.get('relation_type', 'other'), 0.5)
        exploitation = child.score / (child.visits + 1e-6)
        exploration = math.sqrt(2 * math.log(parent.visits + 1) / (child.visits + 1e-6))
        suspense_factor = (child.uncertainty + child.threat_level + child.escape_restriction) / 3
        narrative_bonus = self.narrative_memory.get(child.scene['type'], 0)
        return (exploitation + self.exploration_weight * exploration) * \
               relation_weight * (1 + suspense_factor) * (1 + narrative_bonus * 0.2)

    def expand(self, node):
        unexpanded = [child for child in node.scene.get('children', [])
                     if child not in [n.scene for n in node.children]]
        if not unexpanded:
            return node
        # Calculate the expansion score
        expansion_scores = []
        for child in unexpanded:
            score = self._calculate_expansion_score(child)
            expansion_scores.append((score, child))
        # Softmax
        scores = [s[0] for s in expansion_scores]
        softmax_probs = self._softmax(scores)
        selected_child = random.choices(
            unexpanded,
            weights=softmax_probs,
            k=1
        )[0]
        new_node = SuspenseNode(selected_child, parent=node)
        node.children.append(new_node)
        return new_node

    def _calculate_expansion_score(self, scene):
        priority = scene.get('priority', 0)
        relation_weight = self.relation_weights.get(scene.get('relation_type'), 0.5)
        narrative_value = self.narrative_memory.get(scene['type'], 0)
        branch_factor = len(scene.get('children', [])) * 0.1
        return priority * relation_weight * (1 + narrative_value) + branch_factor

    def _softmax(self, scores):
        exp_scores = [math.exp(s) for s in scores]
        sum_exp_scores = sum(exp_scores)
        return [e / sum_exp_scores for e in exp_scores]

    def simulate(self, node):
        current = node
        path = [current]
        cumulative_score = 0
        depth = 0
        while depth < self.max_depth and not current.is_terminal():
            next_node = self._select_simulation_child(current)
            if next_node is None:
                break
            path.append(next_node)
            current = next_node
            depth += 1
            path_score = self._evaluate_path_suspense(path)
            cumulative_score += path_score
            if self._should_terminate_simulation(path):
                break
        self._update_narrative_memory(path)
        return cumulative_score / len(path)

    def _select_simulation_child(self, node):
        """The selection of child nodes in the simulation phase"""
        if not node.children:
            possible_children = node.scene.get('children', [])
            if not possible_children:
                return None
            weights = []
            for child in possible_children:
                relation_type = child.get('relation_type', 'OTHER')
                if relation_type.upper() == 'TIME':
                    base_weight = 0.6  # Weight for time relationship
                elif relation_type.upper() == 'INFERENCE':
                    base_weight = 0.8  # Higher weight for inference relationship
                else:
                    base_weight = self.relation_weights.get(relation_type, 0.4)

                suspense_score = (
                                         child.get('uncertainty', 0.5) +
                                         child.get('threat_level', 0.5)
                                 ) / 2

                suspense_score *= base_weight  # Incorporate the relationship type weight into the total score

                # Add the calculated score to the weight list
                weights.append(suspense_score)

            total = sum(weights)
            if total == 0:
                return random.choice(possible_children)
            probs = [w / total for w in weights]
            selected = random.choices(possible_children, weights=probs, k=1)[0]
            return SuspenseNode(selected)
        return random.choice(node.children)

    def _check_termination(self, path):
        if len(path) >= self.max_depth:
            return True
        # New endgame condition: At least 3 murderer characteristics appear
        murder_clues = sum(1 for node in path if node.scene.get('murder_related'))
        return murder_clues >= 3  # <source_id>MCTSController.py</source_id>

    def _evaluate_path_suspense(self, path):
        if len(path) < 2:
            return 0
        suspense_score = 0
        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]
            transition_score = (
                curr.uncertainty * 0.4 +
                curr.threat_level * 0.3 +
                curr.escape_restriction * 0.3
            )
            relation_weight = self.relation_weights.get(
                curr.scene.get('relation_type'), 0.5)
            suspense_score += transition_score * relation_weight
        return suspense_score / (len(path) - 1)

    def _should_terminate_simulation(self, path):
        if len(path) < 3:
            return False
        recent_suspense = [n.uncertainty + n.threat_level for n in path[-3:]]
        avg_suspense = sum(recent_suspense) / 3
        return avg_suspense < 0.3

    def _update_narrative_memory(self, path):
        for node in path:
            scene_type = node.scene.get('type')
            self.narrative_memory[scene_type] += 0.1
        for key in self.narrative_memory:
            self.narrative_memory[key] *= 0.95

    def backpropagate(self, node, score, path):
        if any(n.scene.get('id') in [1, 2, 3, 4, 5] for n in path):
            score *= 1.2  # Increase the weight of the endgame scene
        current = node
        depth = len(path)
        current = node
        while current is not None:
            current.visits += 1
            # Suppose there is an update logic for the total_score here.
            current.total_score += score
            current = current.parent


    def _extract_best_path(self):
        path = [self.root]
        current = self.root
        while current.children:
            current = max(current.children,
                         key=lambda c: c.score / c.visits if c.visits > 0 else 0)
            path.append(current)
        return path

    def adjust_weights_dynamically(self, current_path):
        relation_counts = defaultdict(int)
        for node in current_path:
            rel_type = node.scene.get('relation_type', 'other')
            relation_counts[rel_type] += 1
        total_relations = len(current_path)
        for rel_type in self.relation_weights:
            ratio = relation_counts[rel_type] / total_relations
            if ratio < 0.15:
                self.relation_weights[rel_type] *= min(1.2, 1.0 + (0.15 - ratio) * 3)
            else:
                self.relation_weights[rel_type] *= 0.95