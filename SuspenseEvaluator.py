import math
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from Node import SuspenseNode, SuspenseMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('suspense_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SuspenseMetrics:
    """Data class for suspense metrics"""
    uncertainty: float
    threat_level: float
    escape_restriction: float
    narrative_techniques: List[str]
    duration: int


class SuspenseScorer:
    """Suspense scorer"""
    # Configuration part remains unchanged
    WEIGHTS = {
        'uncertainty': 0.3,
        'threat': 0.4,
        'escape': 0.3
    }

    TIME_CONFIG = {
        'normalization_factor': 30,
        'min_duration': 10,
        'max_duration': 60
    }

    NARRATIVE_CONFIG = {
        'technique_multiplier': 0.2,
        'max_techniques': 5
    }

    PENALTY_CONFIG = {
        'continuity_penalty': 0.2,
        'max_penalty_factor': 0.5
    }

    @classmethod
    def calculate_path_suspense(cls, path: List[SuspenseNode]) -> Dict[str, float]:
        """
        Calculate the comprehensive suspense value of the path.
        Now using the unified SuspenseNode class.
        """
        try:
            logger.info(f"开始计算路径悬疑度，路径长度: {len(path)}")

            # Use the metrics attribute of the node
            base_scores = []
            time_factors = []
            narrative_factors = []
            penalties = 0

            # Calculate the average duration
            durations = [node.metrics.duration for node in path]
            avg_duration = sum(durations) / len(durations)

            for i, node in enumerate(path):
                base_scores.append(cls._calculate_base_score(node))
                time_factors.append(cls._calculate_time_factor(node, avg_duration))
                narrative_factors.append(cls._calculate_narrative_factor(node))
                penalties += cls._calculate_penalty(node, i, path)

            final_scores = cls._synthesize_scores(
                base_scores, time_factors, narrative_factors, penalties
            )

            logger.info(f"Path scoring completed, total score: {final_scores['total']}")
            return final_scores

        except Exception as e:
            logger.error(f"Scoring calculation failed: {str(e)}", exc_info=True)
            raise

    @classmethod
    def _calculate_base_score(cls, node: SuspenseNode) -> float:
        """Calculate the base score using the node's metrics"""
        return (
                node.metrics.uncertainty * cls.WEIGHTS['uncertainty'] +
                node.metrics.threat_level * cls.WEIGHTS['threat'] +
                node.metrics.escape_restriction * cls.WEIGHTS['escape']
        )

    @classmethod
    def _calculate_time_factor(cls, node: SuspenseNode, avg_duration: float) -> float:
        """Calculate the time factor"""
        duration = node.metrics.duration

        # Calculate the deviation from the average duration
        deviation = abs(duration - avg_duration)
        normalized_deviation = deviation / cls.TIME_CONFIG['normalization_factor']

        # Use the sigmoid function to smooth the time factor
        return 1 / (1 + math.exp(-normalized_deviation))

    @classmethod
    def _calculate_narrative_factor(cls, node: SuspenseNode) -> float:
        """Calculate the narrative technique factor"""
        technique_count = len(node.metrics.narrative_techniques)
        return 1 + min(
            technique_count * cls.NARRATIVE_CONFIG['technique_multiplier'],
            cls.NARRATIVE_CONFIG['max_techniques'] * cls.NARRATIVE_CONFIG['technique_multiplier']
        )

    @classmethod
    def _calculate_penalty(cls, node: SuspenseNode, index: int, path: List[SuspenseNode]) -> float:
        """Calculate the penalty value"""
        penalty = 0

        # Continuity penalty
        if len(node.children) == 0 and index < len(path) - 1:
            penalty += cls.PENALTY_CONFIG['continuity_penalty']

        # Rhythm penalty
        if index > 0:
            prev_duration = path[index - 1].metrics.duration
            if abs(node.metrics.duration - prev_duration) < cls.TIME_CONFIG['min_duration']:
                penalty += 0.1

        return min(penalty, cls.PENALTY_CONFIG['max_penalty_factor'])

    @classmethod
    def _synthesize_scores(
            cls,
            base_scores: List[float],
            time_factors: List[float],
            narrative_factors: List[float],
            penalties: float
    ) -> Dict[str, float]:
        """Synthesize the final scores"""
        # Calculate the average of each component
        avg_base = sum(base_scores) / len(base_scores)
        avg_time = sum(time_factors) / len(time_factors)
        avg_narrative = sum(narrative_factors) / len(narrative_factors)

        # Calculate the total score after penalty
        penalty_factor = 1 - min(penalties, cls.PENALTY_CONFIG['max_penalty_factor'])
        total_score = avg_base * avg_time * avg_narrative * penalty_factor

        return {
            'total': total_score,
            'base_average': avg_base,
            'time_factor': avg_time,
            'narrative_factor': avg_narrative,
            'penalty_factor': penalty_factor
        }


def test_suspense_scorer():
    """Test function"""
    # Create test scenes
    test_scenes = [
        {
            'id': 'scene1',
            'duration': 40,
            'narrative_techniques': ['flashback', 'suspense_cut'],
            'threats': ['danger1', 'danger2'],
            'available_options': ['option1'],
            'ambiguity': 0.8
        },
        {
            'id': 'scene2',
            'duration': 25,
            'narrative_techniques': ['reversal'],
            'threats': ['danger1'],
            'available_options': ['option1', 'option2'],
            'ambiguity': 0.5
        },
        {
            'id': 'scene3',
            'duration': 30,
            'narrative_techniques': [],
            'threats': [],
            'available_options': ['option1', 'option2', 'option3'],
            'ambiguity': 0.3
        }
    ]

    # Create test nodes
    test_nodes = [SuspenseNode(scene) for scene in test_scenes]

    # Set up child node relationships
    test_nodes[0].children = [test_nodes[1]]
    test_nodes[1].children = [test_nodes[2]]

    # Calculate path suspense
    scores = SuspenseScorer.calculate_path_suspense(test_nodes)

    # Print the results
    print("\Print the results:")
    for key, value in scores.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    test_suspense_scorer()
