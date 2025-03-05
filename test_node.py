from Node import SuspenseNode

test_scene = {
    'id': 'test1',
    'relations': [
        {'type': 'Causation', 'weight': None},  #  Trigger LOW_WEIGHT_CAUSATION
        {'type': 'InvalidType'}  # Trigger INVALID_RELATION
    ]
}
node = SuspenseNode(test_scene)
print(node.metrics.uncertainty)  # Should return base*0.7*0.5 = base*0.35