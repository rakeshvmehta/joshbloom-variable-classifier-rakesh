import time
import tempfile
import os
from collections import defaultdict
from galaxy_hierarchy import GalaxyHierarchy

def test_hierarchy():
    """Comprehensive test suite for GalaxyHierarchy."""
    print("=" * 60)
    print("GALAXY HIERARCHY COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Create a new hierarchy
    hierarchy = GalaxyHierarchy()
    
    # Test 1: Basic Structure Validation
    print("\n1. BASIC STRUCTURE VALIDATION")
    print("-" * 40)
    
    print(f"Root node: {hierarchy.root}")
    print(f"Is tree: {hierarchy.is_tree()}")
    print(f"Total nodes: {len(hierarchy.nodes)}")
    print(f"Max height: {hierarchy.max_height}")
    
    # Validate all classification label nodes are present in the hierarchy
    expected_key_nodes = [
        'amount_of_smoothness_for_disk',
        'signs_of_features_or_disk', 
        'classified_as_star_or_artifact',
        'disk_can_be_viewed_edge_on',
        'disk_cannot_be_viewed_edge_on',
        'bar_or_feature_through_centre_detected',
        'bar_or_feature_through_centre_not_detected',
        'signs_of_spiral_arm_pattern',
        'no_signs_of_spiral_arm_pattern',
        'no_bulge',
        'just_noticeable_bulge',
        'obvious_bulge',
        'dominant_bulge',
        'odd_features_in_image',
        'no_odd_features_in_image',
        'completely_round',
        'in_between_round_and_cigar',
        'cigar_shaped',
        'presence_of_ring',
        'presence_of_lens_or_arc',
        'disturbed_galaxy',
        'irregular_galaxy',
        'presence_of_other_odd_features',
        'merger_galaxies',
        'presence_of_dust_lane',
        'rounded_bulge_at_centre',
        'boxy_bulge_at_centre',
        'no_bulge_at_centre',
        'tight_spiral_arms',
        'medium_spiral_arms',
        'loose_spiral_arms',
        '1_spiral_arm',
        '2_spiral_arms',
        '3_spiral_arms',
        '4_spiral_arms',
        'more_than_4_spiral_arms',
        'cant_tell_if_any_spiral_arms_exist'
    ]
    
    print(f"Expected classification nodes: {len(expected_key_nodes)}")
    missing_nodes = [node for node in expected_key_nodes if node not in hierarchy.nodes]
    if missing_nodes:
        print(f"Missing nodes: {missing_nodes}")
    else:
        print("All 37 classification label nodes present")
    
    # Test 2: Hierarchy Relationships
    print("\n2. HIERARCHY RELATIONSHIPS")
    print("-" * 40)
    
    # Test key relationships based on current hierarchy structure
    test_relationships = [
        ('initial_assessment', 'galaxy'),
        ('odd_features', 'galaxy'),
        ('specific_features', 'galaxy'),
        ('amount_of_smoothness_for_disk', 'initial_assessment'),
        ('signs_of_features_or_disk', 'initial_assessment'),
        ('classified_as_star_or_artifact', 'initial_assessment'),
        ('disk_can_be_viewed_edge_on', 'signs_of_features_or_disk'),
        ('disk_cannot_be_viewed_edge_on', 'signs_of_features_or_disk'),
        ('bar_or_feature_through_centre_detected', 'disk_cannot_be_viewed_edge_on'),
        ('spiral_features', 'disk_cannot_be_viewed_edge_on'),
        ('face_on_bulge', 'spiral_features'),
        ('signs_of_spiral_arm_pattern', 'spiral_features'),
        ('arm_count', 'signs_of_spiral_arm_pattern')
    ]
    
    relationship_errors = []
    for child, expected_parent in test_relationships:
        if child in hierarchy.parents and expected_parent in hierarchy.parents[child]:
            print(f"PASS: {child} -> {expected_parent}")
        else:
            print(f"FAIL: {child} -> {expected_parent}")
            relationship_errors.append((child, expected_parent))
    
    if relationship_errors:
        print(f"FAIL: {len(relationship_errors)} relationship errors found")
    else:
        print("All key relationships validated")
    
    # Test 3: Cycle Detection
    print("\n3. CYCLE DETECTION")
    print("-" * 40)
    
    def has_cycle(hierarchy):
        """Check for cycles in the hierarchy using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for parent in hierarchy.parents[node]:
                if dfs(parent):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in hierarchy.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        return False
    
    if has_cycle(hierarchy):
        print("FAIL: Cycle detected in hierarchy")
    else:
        print("No cycles detected")
    
    # Test 4: Depth and Height Calculations
    print("\n4. DEPTH AND HEIGHT CALCULATIONS")
    print("-" * 40)
    
    # Test depths
    test_nodes = ['galaxy', 'initial_assessment', 'amount_of_smoothness_for_disk', 'spiral_features', 
                  'signs_of_spiral_arm_pattern', 'completely_round', 'tight_spiral_arms']
    
    print("Node depths:")
    for node in test_nodes:
        if node in hierarchy.nodes:
            depth = hierarchy.depth(node)
            print(f"  {node}: {depth}")
        else:
            print(f"  {node}: NODE NOT FOUND")
    
    # Test heights
    print("\nNode heights:")
    for node in test_nodes:
        if node in hierarchy.nodes:
            height = hierarchy.heights[node]
            print(f"  {node}: {height}")
    
    # Test 5: LCS (Lowest Common Subsumer)
    print("\n5. LOWEST COMMON SUBSUMER (LCS)")
    print("-" * 40)
    
    lcs_tests = [
        ('disk_can_be_viewed_edge_on', 'disk_cannot_be_viewed_edge_on'),
        ('bar_or_feature_through_centre_detected', 'bar_or_feature_through_centre_not_detected'),
        ('tight_spiral_arms', 'loose_spiral_arms'),
        ('completely_round', 'cigar_shaped'),
        ('2_spiral_arms', '4_spiral_arms'),
        ('rounded_bulge_at_centre', 'no_bulge_at_centre')
    ]
    
    for node1, node2 in lcs_tests:
        if node1 in hierarchy.nodes and node2 in hierarchy.nodes:
            lcs = hierarchy.lcs(node1, node2)
            print(f"  LCS({node1}, {node2}) = {lcs}")
        else:
            print(f"  LCS({node1}, {node2}) = NODE(S) NOT FOUND")
    
    # Test 6: Wu-Palmer Similarity
    print("\n6. WU-PALMER SIMILARITY")
    print("-" * 40)
    
    for node1, node2 in lcs_tests:
        if node1 in hierarchy.nodes and node2 in hierarchy.nodes:
            wup_sim = hierarchy.wup_similarity(node1, node2)
            print(f"  WuP({node1}, {node2}) = {wup_sim:.3f}")
    
    # Test 7: Custom Metric2
    print("\n7. CUSTOM METRIC2 SIMILARITY")
    print("-" * 40)
    
    for node1, node2 in lcs_tests:
        if node1 in hierarchy.nodes and node2 in hierarchy.nodes:
            metric2_sim = hierarchy.metric2(node1, node2)
            print(f"  Metric2({node1}, {node2}) = {metric2_sim:.3f}")
    
    # Test 8: Edge Cases
    print("\n8. EDGE CASES")
    print("-" * 40)
    
    # Test with non-existent nodes
    try:
        hierarchy.depth('non_existent_node')
        print("FAIL: Should have failed for non-existent node in depth()")
    except KeyError:
        print("Correctly handles non-existent nodes in depth()")
    
    # Test similarity methods with non-existent nodes
    try:
        hierarchy.wup_similarity('non_existent_node1', 'non_existent_node2')
        print("FAIL: Should have failed for non-existent nodes in wup_similarity()")
    except KeyError:
        print("Correctly handles non-existent nodes in wup_similarity()")
    
    try:
        hierarchy.metric2('galaxy', 'non_existent_node')
        print("FAIL: Should have failed for non-existent node in metric2()")
    except KeyError:
        print("Correctly handles non-existent nodes in metric2()")
    
    # Test self-similarity
    if 'galaxy' in hierarchy.nodes:
        self_sim = hierarchy.wup_similarity('galaxy', 'galaxy')
        print(f"  Self-similarity (galaxy, galaxy) = {self_sim:.3f}")
        if abs(self_sim - 1.0) < 1e-6:
            print("Self-similarity is 1.0")
        else:
            print("FAIL: Self-similarity should be 1.0")
    
    # Test 9: Performance and Caching
    print("\n9. PERFORMANCE AND CACHING")
    print("-" * 40)
    
    # Test caching performance
    test_pairs = [
        ('tight_spiral_arms', 'loose_spiral_arms'),
        ('completely_round', 'cigar_shaped'),
        ('2_spiral_arms', '4_spiral_arms')
    ]
    
    # First run (cache miss)
    start_time = time.time()
    for node1, node2 in test_pairs * 100:  # Repeat 100 times
        if node1 in hierarchy.nodes and node2 in hierarchy.nodes:
            _ = hierarchy.wup_similarity(node1, node2)
    first_run_time = time.time() - start_time
    
    # Second run (cache hit)
    start_time = time.time()
    for node1, node2 in test_pairs * 100:  # Repeat 100 times
        if node1 in hierarchy.nodes and node2 in hierarchy.nodes:
            _ = hierarchy.wup_similarity(node1, node2)
    second_run_time = time.time() - start_time
    
    print(f"  First run (cache miss): {first_run_time:.4f}s")
    print(f"  Second run (cache hit): {second_run_time:.4f}s")
    print(f"  Speedup: {first_run_time/second_run_time:.1f}x")
    
    # Test 10: File I/O
    print("\n10. FILE I/O TESTING")
    print("-" * 40)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_filename = f.name
    
    try:
        hierarchy.save(temp_filename)
        print(f"Successfully saved hierarchy to {temp_filename}")
        
        # Load from file
        loaded_hierarchy = GalaxyHierarchy.from_file(temp_filename)
        print(f"Successfully loaded hierarchy from file")
        
        # Compare original and loaded
        if len(hierarchy.nodes) == len(loaded_hierarchy.nodes):
            print("Node count matches")
        else:
            print(f"FAIL: Node count mismatch: {len(hierarchy.nodes)} vs {len(loaded_hierarchy.nodes)}")
        
        # Test a few relationships
        sample_nodes = ['galaxy', 'initial_assessment', 'spiral_features']
        relationships_match = True
        for node in sample_nodes:
            if node in hierarchy.nodes and node in loaded_hierarchy.nodes:
                orig_parents = set(hierarchy.parents[node])
                loaded_parents = set(loaded_hierarchy.parents[node])
                if orig_parents != loaded_parents:
                    relationships_match = False
                    break
        
        if relationships_match:
            print("Relationships preserved in loaded hierarchy")
        else:
            print("FAIL: Relationships not preserved in loaded hierarchy")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
    
    # Test 11: Tree Properties
    print("\n11. TREE PROPERTIES")
    print("-" * 40)
    
    # Check if each node has at most one parent
    multi_parent_nodes = [node for node, parents in hierarchy.parents.items() if len(parents) > 1]
    if multi_parent_nodes:
        print(f"FAIL: Nodes with multiple parents: {multi_parent_nodes[:5]}...")  # Show first 5
        print(f"FAIL: Not a tree structure")
    else:
        print("Each node has at most one parent (tree structure)")
    
    # Check that root has no parents
    if not hierarchy.parents[hierarchy.root]:
        print("Root node has no parents")
    else:
        print(f"FAIL: Root node has parents: {hierarchy.parents[hierarchy.root]}")
    
    # Test 12: Coverage Check
    print("\n12. COVERAGE CHECK")
    print("-" * 40)
    
    # Check that all classification classes are reachable from root
    def get_all_descendants(node, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)
        descendants = {node}
        for child in hierarchy.children[node]:
            descendants.update(get_all_descendants(child, visited))
        return descendants
    
    reachable_nodes = get_all_descendants(hierarchy.root)
    unreachable_nodes = hierarchy.nodes - reachable_nodes
    
    if unreachable_nodes:
        print(f"FAIL: Unreachable nodes: {unreachable_nodes}")
    else:
        print("All nodes reachable from root")
    
    print(f"  Reachable nodes: {len(reachable_nodes)}")
    print(f"  Total nodes: {len(hierarchy.nodes)}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Structure validation: {len(missing_nodes) == 0}")
    print(f"Relationships: {len(relationship_errors) == 0}")
    print(f"No cycles: {not has_cycle(hierarchy)}")
    print(f"Tree structure: {len(multi_parent_nodes) == 0}")
    print(f"All nodes reachable: {len(unreachable_nodes) == 0}")
    print(f"Caching working: {second_run_time < first_run_time}")
    print(f"Total nodes: {len(hierarchy.nodes)}")
    print(f"Key nodes validated: {len(expected_key_nodes)}")
    
    # Show hierarchy statistics
    leaf_nodes = [n for n in hierarchy.nodes if not hierarchy.children[n]]
    intermediate_nodes = [n for n in hierarchy.nodes if hierarchy.children[n] and hierarchy.parents[n]]
    print(f"Leaf nodes: {len(leaf_nodes)}")
    print(f"Intermediate nodes: {len(intermediate_nodes)}")
    print(f"Root nodes: {len([n for n in hierarchy.nodes if not hierarchy.parents[n]])}")
    
    # Display classification labels
    print_classification_labels(hierarchy)

def print_classification_labels(hierarchy):
    """Print the classification labels and their mappings."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION LABELS FOR TRAINING")
    print("=" * 80)
    print(f"Total classification labels: {len(hierarchy.classification_labels)}")
    print("-" * 80)
    
    print("Index | Classification Label | Node Name | Node Type")
    print("-" * 80)
    
    for i, label in enumerate(hierarchy.classification_labels):
        node_name = hierarchy.label_to_node[label]
        if node_name in hierarchy.nodes:
            node_type = "leaf" if not hierarchy.children[node_name] else "intermediate"
            exists = "PASS"
        else:
            node_type = "MISSING"
            exists = "FAIL"
        print(f"{i:5d} | {label:55s} | {node_name:45s} | {node_type}")
    
    print("-" * 80)
    print("Usage examples:")
    print("  # Get index of a label:")
    print("  idx = hierarchy.get_label_index('Completely round')  # Returns index")
    print("  # Get index of a node:")
    print("  idx = hierarchy.get_node_index('completely_round')   # Returns index")
    print("  # Convert prediction index back to label:")
    print("  label = hierarchy.classification_labels[predicted_idx]")
    print("=" * 80)

def print_full_hierarchy(hierarchy):
    """Print the complete hierarchy structure."""
    print("\n" + "=" * 80)
    print("COMPLETE GALAXY MORPHOLOGY HIERARCHY STRUCTURE")
    print("=" * 80)
    print(f"Total nodes: {len(hierarchy.nodes)}")
    print(f"Root: {hierarchy.root}")
    print(f"Max height: {hierarchy.max_height}")
    print("-" * 80)
    
    def print_tree(node, level=0, visited=None, is_last_child=True, prefix=""):
        if visited is None:
            visited = set()
        
        if node in visited:
            return
        visited.add(node)
        
        # Create tree-like visualization
        if level == 0:
            print(f"{node}")
        else:
            # Use tree characters for better visualization
            branch = "└── " if is_last_child else "├── "
            print(f"{prefix}{branch}{node}")
        
        # Sort children for consistent output
        children = sorted(hierarchy.children[node])
        
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            # Update prefix for next level
            if level == 0:
                child_prefix = ""
            else:
                extension = "    " if is_last_child else "│   "
                child_prefix = prefix + extension
            
            print_tree(child, level + 1, visited, is_last, child_prefix)
    
    print_tree(hierarchy.root)
    
    # Print some statistics
    print("\n" + "-" * 80)
    print("HIERARCHY STATISTICS:")
    leaf_nodes = [n for n in hierarchy.nodes if not hierarchy.children[n]]
    intermediate_nodes = [n for n in hierarchy.nodes if hierarchy.children[n] and hierarchy.parents[n]]
    print(f"• Leaf nodes: {len(leaf_nodes)}")
    print(f"• Intermediate nodes: {len(intermediate_nodes)}")
    print(f"• Total nodes: {len(hierarchy.nodes)}")
    print(f"• Average branching factor: {sum(len(children) for children in hierarchy.children.values()) / len([n for n in hierarchy.nodes if hierarchy.children[n]]):.2f}")
    
    # Show some example paths from root to various nodes
    print(f"• Sample paths in hierarchy:")
    def get_path_to_root(node):
        path = [node]
        current = node
        while hierarchy.parents[current]:
            parent = hierarchy.parents[current][0]  # Assuming tree structure
            path.append(parent)
            current = parent
        return list(reversed(path))
    
    # Show a mix of leaf and intermediate nodes
    sample_nodes = sorted(list(hierarchy.nodes))[:3]  # Show first 3 alphabetically
    
    for node in sample_nodes:
        path = get_path_to_root(node)
        path_str = " → ".join(path)
        node_type = "leaf" if not hierarchy.children[node] else "intermediate"
        print(f"    {path_str} ({node_type})")
    
    print("=" * 80)

if __name__ == "__main__":
    test_hierarchy()
    
    # Optionally print the full hierarchy
    print("\nWould you like to see the complete hierarchy structure? (y/n): ", end="")
    user_choice = input().strip().lower()
    
    if user_choice in ['y', 'yes']:
        hierarchy = GalaxyHierarchy()
        print_full_hierarchy(hierarchy)
    else:
        print("Skipping hierarchy visualization.")