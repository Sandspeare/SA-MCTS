class NarrativeIntegrator:
    def __init__(self, graph_connector):
        self.graph = graph_connector
        self.convergence_points = []  # Used to store the convergence points of the two storylines

    def integrate_storylines(self, plot_line, detective_line):
        """
        Integrate the storylines, adding error handling and relationship checking
        Parameters:
         - plot_line: The path of the story occurrence line (allowing 2 to 6 scenes)
         - detective_line: The path of the detective line (allowing 2 to 6 scenes)
        Return the interwoven story path (scene nodes in the form of a list)"""
        if not plot_line or not detective_line:
            raise ValueError("The story path cannot be empty")

        try:
            # Determine the source node and target node for the time sequence association
            # If the number of elements in plot_line is less than 3, use the last one as the starting scene for the time sequence
            time_seq_source = plot_line[2] if len(plot_line) > 2 else plot_line[-1]
            # Check and create the time sequence association
            if not self._check_existing_relation(
                    time_seq_source.scene['id'],
                    detective_line[0].scene['id'],
                    'TIME_SEQUENCE'
            ):
                self._create_time_sequence(time_seq_source, detective_line[0])

            # Determine the source node and target node for the causal association
            # If detective_line has at least 2 scenes, use the second last scene; otherwise, use the first scene
            detective_causal_source = detective_line[-2] if len(detective_line) > 1 else detective_line[0]
            if not self._check_existing_relation(
                    plot_line[-1].scene['id'],
                    detective_causal_source.scene['id'],
                    'CAUSES'
            ):
                self._create_causal_relation(plot_line[-1], detective_causal_source)

            # Handle the key clue association (take the first min(3, len(detective_line)) scenes of detective_line)
            # For the target scene, take the 4th scene in plot_line or the last one if there are not enough scenes
            target_index = 3 if len(plot_line) > 3 else -1
            for clue_node in detective_line[:min(3, len(detective_line))]:
                if "关键证据" in clue_node.scene.get('tags', []):
                    if not self._check_existing_relation(
                            clue_node.scene['id'],
                            plot_line[target_index].scene['id'],
                            'CAUSES'
                    ):
                        self._mark_explicit_causality(clue_node, plot_line[target_index])

            # Interweave the storylines: Use a simple alternating merge method for integration
            integrated_path = []
            max_len = max(len(plot_line), len(detective_line))
            for i in range(max_len):
                if i < len(plot_line):
                    integrated_path.append(plot_line[i])
                if i < len(detective_line):
                    integrated_path.append(detective_line[i])

            # Check the characteristics of the endgame scene. If they do not meet the requirements, create a manual convergence
            if integrated_path and not self._check_murder_convergence(integrated_path[-1]):
                integrated_path = self._create_manual_convergence(integrated_path)

            return integrated_path

        except Exception as e:
            print(f"Storyline integration failed:  {str(e)}")
            raise

    def _create_time_sequence(self, source_node, target_node):
        """Create a time sequence association"""
        query = """
        MATCH (source:Scene {id: $source_id})
        MATCH (target:Scene {id: $target_id})
        WITH source, target
        WHERE source IS NOT NULL AND target IS NOT NULL
        CREATE (source)-[:TIME_SEQUENCE {type: 'synchronous'}]->(target)
        """
        try:
            self.graph.execute_query(
                query,
                {'source_id': source_node.scene['id'],
                 'target_id': target_node.scene['id']}
            )
        except Exception as e:
            print(f"Failed to create the time sequence association:{str(e)}")

    def _create_causal_relation(self, source_node, target_node):
        """Create a causal association"""
        query = """
        MATCH (source:Scene {id: $source_id})
        MATCH (target:Scene {id: $target_id})
        WITH source, target
        WHERE source IS NOT NULL AND target IS NOT NULL
        CREATE (source)-[:CAUSES {
            type: 'explicit',
            weight: 1.0,
            timestamp: datetime()
        }]->(target)
        """
        try:
            self.graph.execute_query(
                query,
                {'source_id': source_node.scene['id'],
                 'target_id': target_node.scene['id']}
            )
        except Exception as e:
            print(f"Failed to create the causal association:  {str(e)}")

    def _mark_explicit_causality(self, clue_node, target_node):
        """Mark the explicit causal relationship"""
        query = """
        MATCH (source:Scene {id: $source_id})
        MATCH (target:Scene {id: $target_id})
        WITH source, target
        WHERE source IS NOT NULL AND target IS NOT NULL
        MERGE (source)-[r:CAUSES {
            type: 'explicit',
            note: 'Key clue association'
        }]->(target)
        ON CREATE SET r.created = datetime()
        """
        try:
            self.graph.execute_query(
                query,
                {'source_id': clue_node.scene['id'],
                 'target_id': target_node.scene['id']}
            )
        except Exception as e:
            print(f"Failed to mark the explicit causal relationship: {str(e)}")

    def _check_murder_convergence(self, final_node):
        """Check whether the endgame scene meets the convergence conditions"""
        required_features = {
            'taxi': False,
            'poison': False,
            'rache': False,
            'appearance': False
        }

        # Check the clues
        clues = final_node.scene.get('clues', [])
        for clue in clues:
            clue_lower = clue.lower()
            if 'taxi' in clue_lower:
                required_features['taxi'] = True
            elif 'poison' in clue_lower:
                required_features['poison'] = True
            elif 'rache' in clue_lower:
                required_features['rache'] = True
            elif any(feature in clue_lower for feature in ['高大', '脸色', '大衣']):
                required_features['appearance'] = True

        return all(required_features.values())

    def _create_manual_convergence(self, path):
        """Create a manually converged endgame scene"""
        final_scene = {
            'id': 'final_convergence',
            'type': 'revelation',
            'clues': ['taxi driver', 'poisoning', 'rival in love', 'RACHE', 'tall', 'ruddy complexion', 'brown coat'],
            'is_ending': True,
            'culprit_linked': True,
            'detective_verify': True
        }

        from Node import SuspenseNode
        convergence_node = SuspenseNode(final_scene)

        path.append(convergence_node)
        return path

    def _is_suitable_insertion_point(self, current_node, detective_node):
        """Determine whether it is suitable to insert the detective scene at the current position"""
        return (self._check_timeline_compatibility(current_node, detective_node) and
                self._check_location_compatibility(current_node, detective_node) and
                self._check_narrative_coherence(current_node, detective_node))

    def _check_timeline_compatibility(self, scene1, scene2):
        """Check the timeline compatibility"""
        time1 = scene1.scene.get('timestamp', 0)
        time2 = scene2.scene.get('timestamp', 0)
        return abs(time1 - time2) <= 2

    def _check_existing_relation(self, source_id, target_id, relation_type):
        """Check whether there is an existing relation of the specified type between two scenes"""
        query = """
        MATCH (source:Scene {id: $source_id})
        MATCH (target:Scene {id: $target_id})
        WITH source, target
        MATCH (source)-[r]->(target)
        WHERE type(r) = $relation_type
        RETURN count(r) > 0 as exists
        """
        try:
            result = self.graph.execute_query(
                query,
                {
                    'source_id': source_id,
                    'target_id': target_id,
                    'relation_type': relation_type
                }
            )
            return result[0]['exists'] if result else False
        except Exception as e:
            print(f"检查关系存在性失败: {str(e)}")
            return False

    def _check_location_compatibility(self, scene1, scene2):
        """Check the spatial location compatibility"""
        loc1 = scene1.scene.get('location', '')
        loc2 = scene2.scene.get('location', '')
        return loc1 == loc2 or loc1 == '' or loc2 == ''

    def _check_narrative_coherence(self, scene1, scene2):
        """Check the narrative coherence"""
        type1 = scene1.scene.get('type', '')
        type2 = scene2.scene.get('type', '')

        valid_transitions = {
            'action': ['investigation', 'dialogue', 'revelation'],
            'investigation': ['dialogue', 'action', 'clue'],
            'dialogue': ['action', 'investigation', 'revelation'],
            'clue': ['investigation', 'dialogue'],
            'revelation': ['action', 'dialogue']
        }

        return type2 in valid_transitions.get(type1, [])

    def _check_character_connection(self, scene1, scene2):
        """Check the character connection"""
        chars1 = set(scene1.scene.get('characters', []))
        chars2 = set(scene2.scene.get('characters', []))
        return bool(chars1.intersection(chars2))

    def _evaluate_unexpected_meeting(self, scene1, scene2):
        """Evaluate the degree of unexpected meeting"""
        score = 0.5
        if scene1.get('type') != scene2.get('type'):
            score += 0.2
        common_chars = set(scene1.get('characters', [])) & set(scene2.get('characters', []))
        if common_chars:
            score -= 0.1 * len(common_chars)
        return max(0, min(score, 1.0))

    def _evaluate_dramatic_conflict(self, scene1, scene2):
        """Evaluate the degree of dramatic conflict"""
        score = 0.0
        conflict_keywords = {'confrontation', 'revelation', 'crisis', 'decision'}
        scene1_tags = set(scene1.get('tags', []))
        scene2_tags = set(scene2.get('tags', []))
        conflict_count = len((scene1_tags | scene2_tags) & conflict_keywords)
        score += 0.2 * conflict_count
        return min(score, 1.0)

    def _evaluate_information_revelation(self, scene1, scene2):
        """Evaluate the degree of information revelation"""
        score = 0.0
        if scene1.get('clues') or scene2.get('clues'):
            score += 0.3
        if scene1.get('culprit_related') or scene2.get('culprit_related'):
            score += 0.4
        if scene1.get('new_information') or scene2.get('new_information'):
            score += 0.3
        return min(score, 1.0)