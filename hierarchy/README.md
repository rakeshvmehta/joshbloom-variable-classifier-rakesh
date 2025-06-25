# Galaxy Morphology Hierarchy

This module provides a hierarchical classification system for galaxy morphology, based on the Galaxy Zoo project's classification scheme. The hierarchy captures the relationships between different types of galaxies and provides tools for computing semantic similarities between classes.

## Hierarchy Structure

The hierarchy is organized as follows:

```
galaxy
├── elliptical
│   ├── round
│   ├── cigar_shaped
│   └── boxy
├── spiral
│   ├── barred
│   │   ├── strong_bar
│   │   └── weak_bar
│   └── unbarred
│       ├── grand_design
│       └── flocculent
├── lenticular
│   ├── edge_on
│   └── face_on
├── irregular
│   ├── dwarf
│   └── interacting
└── special_cases
    ├── ring
    ├── polar_ring
    ├── merger
    └── peculiar
```

## Features

- Hierarchical representation of galaxy morphology
- Semantic similarity computation between classes
- Lowest Common Subsumer (LCS) finding
- Depth and height calculations
- File-based persistence
- Performance-optimized with caching

## Usage

### Basic Usage

```python
from galaxy_hierarchy import GalaxyHierarchy

# Create a new hierarchy
hierarchy = GalaxyHierarchy()

# Get similarity between two classes
similarity = hierarchy.metric2('spiral', 'elliptical')

# Find the lowest common subsumer
lcs = hierarchy.lcs('barred', 'unbarred')

# Get the depth of a class
depth = hierarchy.depth('strong_bar')
```

### Saving and Loading

```python
# Save the hierarchy to a file
hierarchy.save('galaxy_hierarchy.txt')

# Load the hierarchy from a file
loaded_hierarchy = GalaxyHierarchy.from_file('galaxy_hierarchy.txt')
```

### Testing

Run the test file to see examples of all functionality:

```bash
python test_hierarchy.py
```

## Integration with CNN

To integrate this hierarchy with your CNN:

1. Use the hierarchy to compute semantic similarities between classes
2. Implement a hierarchical loss function that considers class relationships
3. Use the hierarchy for evaluation metrics that consider semantic similarity
4. Map your CNN's output to the hierarchy's classes
