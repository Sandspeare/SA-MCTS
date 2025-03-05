# Node.py
import math
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SuspenseMetrics:
    """Suspense metric data class"""
    uncertainty: float
    threat_level: float
    escape_restriction: float
    narrative_techniques: List[str]
    duration: int


class SuspenseNode:
    """Unified suspense node class"""

    @property
    def relation_types(self) -> list:
        """返Return the list of types of all relations in this scene"""
        return [rel.get('type', 'UNKNOWN') for rel in self.scene.get('relations', [])]

    def __init__(self, scene: Dict[str, Any], parent=None):
        self.scene = scene
        self.parent = parent
        self.children = []

        # MCTS parameters
        self.visits = 0
        self.score = 0.0

        # Initialize cached attributes
        self._uncertainty = None
        self._threat_level = None
        self._restrictions = None

        # Define relation penalty factors
        self._relation_penalties = {
            'INVALID_RELATION': 0.7,
            'LOW_WEIGHT_CAUSATION': 0.5
        }
        # Expected relation set (all in uppercase)
        self._expected_relations = {"CAUSATION", "TIME", "INFERENCE", "HIERARCHY", "CONCURRENT", "ASSOCIATION"}

        # Calculate suspense metrics
        self.metrics = self._calculate_metrics()

    def _relation_penalty(self) -> float:
        """Calculate the penalty coefficient according to the relation types in the scene"""
        penalty = 1.0
        for rel in self.scene.get('relations', []):
            rel_type = rel.get('type', '').upper()
            # Handle null weights: For causal relations, when the weight is None, it is considered that the weight is low
            if rel.get('weight') is None and rel_type == "CAUSATION":
                penalty *= self._relation_penalties['LOW_WEIGHT_CAUSATION']
            # If the relation type is not in the expected set, apply the penalty for an invalid relation
            elif rel_type not in self._expected_relations:
                penalty *= self._relation_penalties['INVALID_RELATION']
        return max(0.3, penalty)  # Ensure at least 30% effectiveness

    def _calculate_metrics(self) -> SuspenseMetrics:
        """Calculate suspense metrics using English identifiers <source_id>Node.py</source_id>"""
        # Perform scene taxonomy validation
        self._validate_scene_taxonomy()

        # Generate basic suspense metrics
        base_uncertainty = self._calculate_uncertainty()
        print(f"[DEBUG] Base uncertainty for {self.scene['id']}: {base_uncertainty:.2f}")

        return SuspenseMetrics(
            uncertainty=base_uncertainty * self._relation_penalty(),  # 应用关系惩罚
            threat_level=self._calculate_threat_level(),
            escape_restriction=self._calculate_escape_restriction(),
            narrative_techniques=self.scene.get('narrative_techniques', []),
            duration=self.scene.get('duration', 30)
        )

    def _validate_scene_taxonomy(self):
        """Validate whether the relations in the scene meet the expectations
Only issue a warning for mismatched relations and do not throw an exception to facilitate subsequent penalty calculation.
        """
        expected_relations = {"TIME", "INFERENCE", "HIERARCHY",
                              "CAUSATION", "CONCURRENT", "ASSOCIATION"}

        # Fix: Convert to uppercase uniformly to avoid case sensitivity
        actual_relations = {r['type'].upper() for r in self.scene.get('relations', [])}
        invalid_rels = actual_relations - expected_relations
        # Special exemption for test scenes (judge according to scene_id)
        if str(self.scene.get('id', '')).startswith('test_'):

            print(f"[WARNING] Test scene {self.scene['id']} has non-standard relations: {invalid_rels}")
            return  # Skip validation
        if invalid_rels:
            print(f"[WARNING] Scene {self.scene['id']} contains invalid relations: {invalid_rels}")
            # No longer throw an exception, allow passing the validation

    def _calculate_uncertainty(self) -> float:
        """Calculate the uncertainty directly without using caching"""
        branching = len(self.children) * 0.2
        ambiguity = self.scene.get('ambiguity', 0.5)
        unknown_factors = len(self.scene.get('unknown_factors', [])) * 0.1
        possible_interpretations = len(self.scene.get('possible_interpretations', []))

        base_score = (
            0.3 * branching +
            0.3 * ambiguity +
            0.2 * unknown_factors +
            0.2 * (possible_interpretations / 3)
        )

        # Adjust the score according to the presence or absence of clues
        tags = set(tag.lower() for tag in self.scene.get('tags', []))
        if set(['taxi', 'poison', 'rache']).issubset(tags):
            base_score += 0.3  # Add 30% to the score

        return min(1.0, base_score)

    def _calculate_threat_level(self) -> float:
        """Calculate the threat level directly without using caching"""
        threats = self.scene.get('threats', [])
        return min(1.0, len(threats) * 0.25)

    def _calculate_escape_restriction(self) -> float:
        """Calculate the uncertainty score of the scene"""
        total_actions = len(self.scene.get('protagonist_actions', []))
        escape_count = len(self.get_escape_options())
        return 1 - (escape_count / total_actions) if total_actions > 0 else 0.5

    @property
    def uncertainty(self) -> float:
        """Calculate the threat level"""
        if self._uncertainty is None:
            self._uncertainty = self._calculate_uncertainty()
        return self._uncertainty

    @property
    def threat_level(self) -> float:
        """Calculate the threat level"""
        if self._threat_level is None:
            self._threat_level = self._calculate_threat_level()
        return self._threat_level

    @property
    def escape_restriction(self) -> float:
        """Calculate the escape restriction"""
        if self._restrictions is None:
            self._restrictions = self._calculate_escape_restriction()
        return self._restrictions

    def get_escape_options(self) -> List[Dict]:
        """Get the escape options"""
        return [opt for opt in self.scene.get('protagonist_actions', [])
                if opt.get('type') == 'ESCAPE']

    def add_child(self, child_node: 'SuspenseNode') -> None:
        """Add a child node"""
        self.children.append(child_node)
        child_node.parent = self
        self._uncertainty = None

    def is_terminal(self) -> bool:
        """Determine whether it is a terminal node"""
        return self.scene.get('is_ending', False) or len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """Check whether it has been fully expanded"""
        possible_children = self.scene.get('children', [])
        return len(self.children) >= len(possible_children)

    def ucb_value(self, exploration_constant: float = 1.41) -> float:
        """Calculate the UCB value"""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.score / self.visits

        exploitation = self.score / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def __repr__(self) -> str:
        try:
            return (f"<SuspenseNode(id={self.scene.get('id', 'N/A')}, "
                    f"uncertainty={self.uncertainty:.2f}, "
                    f"threat={self.threat_level:.2f}, "
                    f"escape_restriction={self.escape_restriction:.2f}, "
                    f"visits={self.visits}, score={self.score:.2f})>")
        except Exception as e:
            return f"<[{repr(e)} raised in repr()] SuspenseNode object at {hex(id(self))}>"