import numpy as np
from collections import defaultdict

class GalaxyHierarchy:
    """Represents a hierarchical classification system for galaxy morphology."""
    
    def __init__(self):
        # Initialize the hierarchy structure
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        self.nodes = set()
        
        # Initialize caches for performance
        self._depths = {False: {}, True: {}}
        self._hyp_depth_cache = {False: {}, True: {}}
        self._hyp_dist_cache = {}
        self._lcs_cache = {}
        self._wup_cache = {}
        
        # Build the hierarchy
        self._build_hierarchy()
        
        # Define classification labels for training (maps to node names)
        self.classification_labels = [
            'Amount of Smoothness for a disk',
            'Signs of features or a disk', 
            'Classified as Star or Artifact',
            'Disk can be viewed edge-on',
            'Disk cannot be viewed edge-on',
            'Bar or feature through centre of galaxy detected',
            'Bar or feautre through centre of galaxy not detected',
            'Signs of a spiral arm pattern',
            'No signs of a spiral arm pattern',
            'No bulge',
            'Just noticeable bulge',
            'Obvious bulge',
            'Dominant bulge',
            'Odd features in image',
            'No odd features in image',
            'Completely round',
            'In-between round and cigar',
            'Cigar-shaped',
            'Presence of Ring',
            'Presence of lens or arc',
            'Disturbed galaxy',
            'Irregular galaxy',
            'Presence of other odd features',
            'Merger galaxies',
            'Presence of dust lane',
            'Rounded bulge at centre',
            'Boxy bulge at centre',
            'No bulge at centre',
            'Tight spiral arms',
            'Medium spiral arms',
            'Loose spiral arms',
            '1 spiral arm',
            '2 spiral arms',
            '3 spiral arms',
            '4 spiral arms',
            'More than 4 spiral arms',
            'Can\'t tell if any spiral arms exist'
        ]
        
        # Map classification labels to node names
        self.label_to_node = {
            'Amount of Smoothness for a disk': 'amount_of_smoothness_for_disk',
            'Signs of features or a disk': 'signs_of_features_or_disk',
            'Classified as Star or Artifact': 'classified_as_star_or_artifact',
            'Disk can be viewed edge-on': 'disk_can_be_viewed_edge_on',
            'Disk cannot be viewed edge-on': 'disk_cannot_be_viewed_edge_on',
            'Bar or feature through centre of galaxy detected': 'bar_or_feature_through_centre_detected',
            'Bar or feautre through centre of galaxy not detected': 'bar_or_feature_through_centre_not_detected',
            'Signs of a spiral arm pattern': 'signs_of_spiral_arm_pattern',
            'No signs of a spiral arm pattern': 'no_signs_of_spiral_arm_pattern',
            'No bulge': 'no_bulge',
            'Just noticeable bulge': 'just_noticeable_bulge',
            'Obvious bulge': 'obvious_bulge',
            'Dominant bulge': 'dominant_bulge',
            'Odd features in image': 'odd_features_in_image',
            'No odd features in image': 'no_odd_features_in_image',
            'Completely round': 'completely_round',
            'In-between round and cigar': 'in_between_round_and_cigar',
            'Cigar-shaped': 'cigar_shaped',
            'Presence of Ring': 'presence_of_ring',
            'Presence of lens or arc': 'presence_of_lens_or_arc',
            'Disturbed galaxy': 'disturbed_galaxy',
            'Irregular galaxy': 'irregular_galaxy',
            'Presence of other odd features': 'presence_of_other_odd_features',
            'Merger galaxies': 'merger_galaxies',
            'Presence of dust lane': 'presence_of_dust_lane',
            'Rounded bulge at centre': 'rounded_bulge_at_centre',
            'Boxy bulge at centre': 'boxy_bulge_at_centre',
            'No bulge at centre': 'no_bulge_at_centre',
            'Tight spiral arms': 'tight_spiral_arms',
            'Medium spiral arms': 'medium_spiral_arms',
            'Loose spiral arms': 'loose_spiral_arms',
            '1 spiral arm': '1_spiral_arm',
            '2 spiral arms': '2_spiral_arms',
            '3 spiral arms': '3_spiral_arms',
            '4 spiral arms': '4_spiral_arms',
            'More than 4 spiral arms': 'more_than_4_spiral_arms',
            'Can\'t tell if any spiral arms exist': 'cant_tell_if_any_spiral_arms_exist'
        }
        
        # Reverse mapping for convenience
        self.node_to_label = {v: k for k, v in self.label_to_node.items()}
        
        # Compute derived properties
        self._compute_heights()
        self._compute_distances()
        self.max_height = max(self.heights.values())
        self._compute_root()
    
    def _build_hierarchy(self):
        """Build the galaxy morphology hierarchy with all 37 classification classes."""
        # Root node
        self._add_relation('galaxy', None)
        
        # Main assessment categories
        self._add_relation('initial_assessment', 'galaxy')
        self._add_relation('odd_features', 'galaxy')
        self._add_relation('specific_features', 'galaxy')
        
        # Class 1: Initial morphology assessment (3 classes)
        self._add_relation('amount_of_smoothness_for_disk', 'initial_assessment')      # Class1.1
        self._add_relation('signs_of_features_or_disk', 'initial_assessment')          # Class1.2
        self._add_relation('classified_as_star_or_artifact', 'initial_assessment')     # Class1.3
        
        # Class 2: Disk orientation (2 classes)
        self._add_relation('disk_can_be_viewed_edge_on', 'signs_of_features_or_disk')            # Class2.1
        self._add_relation('disk_cannot_be_viewed_edge_on', 'signs_of_features_or_disk')         # Class2.2

        # Class 3: Bar detection (2 classes)
        self._add_relation('bar_or_feature_through_centre_detected', 'disk_cannot_be_viewed_edge_on')     # Class3.1
        self._add_relation('bar_or_feature_through_centre_not_detected', 'disk_cannot_be_viewed_edge_on') # Class3.2
        
        # Class 4: Spiral detection (2 classes)
        self._add_relation('signs_of_spiral_arm_pattern', 'spiral_features')           # Class4.1
        self._add_relation('no_signs_of_spiral_arm_pattern', 'spiral_features')        # Class4.2
        self._add_relation('spiral_features', 'disk_cannot_be_viewed_edge_on')         # spiral_features comes after disk_cannot_be_viewed_edge_on regardless of the value of if bar in center is detected or not (Class3.1)
        
        # Class 5: Edge-on bulge prominence (4 classes)
        self._add_relation('no_bulge', 'disk_can_be_viewed_edge_on')                   # Class5.1
        self._add_relation('just_noticeable_bulge', 'disk_can_be_viewed_edge_on')      # Class5.2
        self._add_relation('obvious_bulge', 'disk_can_be_viewed_edge_on')              # Class5.3
        self._add_relation('dominant_bulge', 'disk_can_be_viewed_edge_on')             # Class5.4
        
        # Class 6: Odd features detection (2 classes)
        self._add_relation('odd_features_in_image', 'odd_features')                    # Class6.1
        self._add_relation('no_odd_features_in_image', 'odd_features')                 # Class6.2
        
        # Class 7: Smooth galaxy shapes (3 classes)
        self._add_relation('completely_round', 'amount_of_smoothness_for_disk')                        # Class7.1
        self._add_relation('in_between_round_and_cigar', 'amount_of_smoothness_for_disk')              # Class7.2
        self._add_relation('cigar_shaped', 'amount_of_smoothness_for_disk')                            # Class7.3
        
        # Class 8: Specific odd features (7 classes)
        self._add_relation('presence_of_ring', 'specific_features')                    # Class8.1
        self._add_relation('presence_of_lens_or_arc', 'specific_features')             # Class8.2
        self._add_relation('disturbed_galaxy', 'specific_features')                    # Class8.3
        self._add_relation('irregular_galaxy', 'specific_features')                    # Class8.4
        self._add_relation('presence_of_other_odd_features', 'specific_features')      # Class8.5
        self._add_relation('merger_galaxies', 'specific_features')                     # Class8.6
        self._add_relation('presence_of_dust_lane', 'specific_features')               # Class8.7
        
        # Class 9: Face-on bulge characteristics (3 classes)
        self._add_relation('rounded_bulge_at_centre', 'face_on_bulge')                 # Class9.1
        self._add_relation('boxy_bulge_at_centre', 'face_on_bulge')                    # Class9.2
        self._add_relation('no_bulge_at_centre', 'face_on_bulge')                      # Class9.3
        self._add_relation('face_on_bulge', 'spiral_features')                         # face_on_bulge comes after spiral_features regardless of if there are signs of spiral arm pattern or not (Class4.1 or Class4.2)
        
        # Class 10: Spiral arm tightness (3 classes)
        self._add_relation('tight_spiral_arms', 'signs_of_spiral_arm_pattern')         # Class10.1
        self._add_relation('medium_spiral_arms', 'signs_of_spiral_arm_pattern')        # Class10.2
        self._add_relation('loose_spiral_arms', 'signs_of_spiral_arm_pattern')         # Class10.3
        
        # Class 11: Number of spiral arms (6 classes)
        self._add_relation('1_spiral_arm', 'arm_count')                                # Class11.1
        self._add_relation('2_spiral_arms', 'arm_count')                               # Class11.2
        self._add_relation('3_spiral_arms', 'arm_count')                               # Class11.3
        self._add_relation('4_spiral_arms', 'arm_count')                               # Class11.4
        self._add_relation('more_than_4_spiral_arms', 'arm_count')                     # Class11.5
        self._add_relation('cant_tell_if_any_spiral_arms_exist', 'arm_count')          # Class11.6
        self._add_relation('arm_count', 'signs_of_spiral_arm_pattern')                 # arm_count comes after signs_of_spiral_arm_pattern (Class10.3)
    
    def _add_relation(self, child, parent):
        """Add a parent-child relationship to the hierarchy."""
        if parent is not None:
            self.parents[child].append(parent)
            self.children[parent].append(child)
        self.nodes.add(child)
        if parent is not None:
            self.nodes.add(parent)
    
    def _compute_heights(self):
        """Compute the height of each node in the hierarchy."""
        def height(id):
            if id not in self.heights:
                self.heights[id] = 1 + max((height(child) for child in self.children[id]), default=-1) if id in self.children else 0
            return self.heights[id]
        
        self.heights = {}
        for node in self.nodes:
            height(node)
    
    def _compute_distances(self):
        """Compute distances between nodes in the hierarchy."""
        def distance(id, id2, pathlength):
            if (id, id2) not in self._hyp_dist_cache:
                if id == id2:
                    self._hyp_dist_cache[(id, id2)] = pathlength
                else:
                    self._hyp_dist_cache[(id, id2)] = min(
                        [distance(p, id2, pathlength + 1) for p in self.parents[id]],
                        default=float('inf')
                    )
            return self._hyp_dist_cache[(id, id2)]
        
        for node in self.nodes:
            for node2 in self.nodes:
                distance(node, node2, 0)
    
    def _compute_root(self):
        """Compute the root node of the hierarchy."""
        self.root = next(node for node in self.nodes if not self.parents[node])
    
    def is_tree(self):
        """Check if the hierarchy is a tree (each node has at most one parent)."""
        return all(len(parents) <= 1 for parents in self.parents.values())
    
    def get_label_index(self, label):
        """Get the index of a classification label."""
        try:
            return self.classification_labels.index(label)
        except ValueError:
            raise ValueError(f"Label '{label}' not found in classification_labels")
    
    def get_node_index(self, node_name):
        """Get the index of a node by converting it to its label first."""
        if node_name in self.node_to_label:
            label = self.node_to_label[node_name]
            return self.get_label_index(label)
        else:
            raise ValueError(f"Node '{node_name}' not found in hierarchy or not mapped to a classification label")
    
    def all_hypernym_depths(self, id, use_min_depth=False):
        """Get all hypernym depths for a node."""
        # Check if node exists
        if id not in self.nodes:
            raise KeyError(f"Node not found in hierarchy: {id}")
        
        if (id, use_min_depth) not in self._hyp_depth_cache[use_min_depth]:
            if not self.parents[id]:
                self._hyp_depth_cache[use_min_depth][(id, use_min_depth)] = {id: 0}
            else:
                depths = {}
                for parent in self.parents[id]:
                    parent_depths = self.all_hypernym_depths(parent, use_min_depth)
                    for node, depth in parent_depths.items():
                        if node not in depths or (use_min_depth and depth < depths[node]) or (not use_min_depth and depth > depths[node]):
                            depths[node] = depth + 1
                depths[id] = 0
                self._hyp_depth_cache[use_min_depth][(id, use_min_depth)] = depths
        return self._hyp_depth_cache[use_min_depth][(id, use_min_depth)]
    
    def lcs(self, a, b, use_min_depth=False):
        """Find the lowest common subsumer of two nodes."""
        # Check if nodes exist
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"One or both nodes not found in hierarchy: {a}, {b}")
        
        if (a, b, use_min_depth) not in self._lcs_cache:
            a_depths = self.all_hypernym_depths(a, use_min_depth)
            b_depths = self.all_hypernym_depths(b, use_min_depth)
            common = set(a_depths.keys()) & set(b_depths.keys())
            if not common:
                self._lcs_cache[(a, b, use_min_depth)] = None
            else:
                self._lcs_cache[(a, b, use_min_depth)] = max(
                    common,
                    key=lambda x: a_depths[x] if use_min_depth else -a_depths[x]
                )
        return self._lcs_cache[(a, b, use_min_depth)]
    
    def wup_similarity(self, a, b):
        """Compute Wu-Palmer similarity between two nodes."""
        # Check if nodes exist
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"One or both nodes not found in hierarchy: {a}, {b}")
        
        if (a, b) not in self._wup_cache:
            # Handle self-similarity case
            if a == b:
                self._wup_cache[(a, b)] = 1.0
            else:
                lcs_node = self.lcs(a, b)
                if lcs_node is None:
                    self._wup_cache[(a, b)] = 0
                else:
                    depth_lcs = self.depth(lcs_node)
                    depth_a = self.depth(a)
                    depth_b = self.depth(b)
                    # Handle division by zero case (when both nodes are at root level)
                    if depth_a + depth_b == 0:
                        self._wup_cache[(a, b)] = 1.0  # Both at root, maximally similar
                    else:
                        self._wup_cache[(a, b)] = 2 * depth_lcs / (depth_a + depth_b)
        return self._wup_cache[(a, b)]
    
    def depth(self, id, use_min_depth=False):
        """Get the depth of a node in the hierarchy."""
        # Check if node exists
        if id not in self.nodes:
            raise KeyError(f"Node not found in hierarchy: {id}")
        
        if (id, use_min_depth) not in self._depths[use_min_depth]:
            if not self.parents[id]:
                self._depths[use_min_depth][(id, use_min_depth)] = 0
            else:
                self._depths[use_min_depth][(id, use_min_depth)] = 1 + max(
                    self.depth(parent, use_min_depth) for parent in self.parents[id]
                )
        return self._depths[use_min_depth][(id, use_min_depth)]
    
    def metric2(self, a, b):
        """Compute a custom similarity metric between two nodes."""
        # Check if nodes exist
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"One or both nodes not found in hierarchy: {a}, {b}")
        
        # Handle self-similarity case
        if a == b:
            return 1.0
        
        lcs_node = self.lcs(a, b)
        if lcs_node is None:
            return 0
        depth_lcs = self.depth(lcs_node)
        depth_a = self.depth(a)
        depth_b = self.depth(b)
        
        # Handle division by zero case (when both nodes are at root level)
        if depth_a + depth_b == 0:
            return 1.0  # Both at root, maximally similar
        else:
            return 2 * depth_lcs / (depth_a + depth_b)
    
    def save(self, filename):
        """Save the hierarchy to a file."""
        with open(filename, 'w') as f:
            for node in sorted(self.nodes):
                parents = self.parents[node]
                if parents:
                    f.write(f"{node} {' '.join(parents)}\n")
    
    @classmethod
    def from_file(cls, filename):
        """Load a hierarchy from a file."""
        hierarchy = cls()
        with open(filename) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    child = parts[0]
                    parents = parts[1:]
                    for parent in parents:
                        hierarchy._add_relation(child, parent)
        return hierarchy 