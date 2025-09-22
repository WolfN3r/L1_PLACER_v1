import json
import sys
import random
import copy
import math
import time


class OptimizationBlock:
    """Represents a block for optimization with position and variant information."""

    def __init__(self, name):
        self.name = name
        self.width = 0.0
        self.height = 0.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.device_type = ""
        self.variants = []
        self.current_variant = None
        self.variant_index = 0
        self.column_multiple = 1
        self.row_multiple = 1
        self.color = "FFFFFF"

    def get_width(self):
        """Get the actual width of the block."""
        return self.x_max - self.x_min if self.x_max > self.x_min else self.width

    def get_height(self):
        """Get the actual height of the block."""
        return self.y_max - self.y_min if self.y_max > self.y_min else self.height

    def get_center_x(self):
        """Get the center X coordinate of the block."""
        return (self.x_min + self.x_max) * 0.5

    def get_center_y(self):
        """Get the center Y coordinate of the block."""
        return (self.y_min + self.y_max) * 0.5

    def set_position(self, x_min, y_min):
        """Set the position of the block."""
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_min + self.width
        self.y_max = y_min + self.height

    def set_variant(self, variant, variant_index=0):
        """Set the variant for this block."""
        if variant:
            self.width = variant['width']
            self.height = variant['height']
            self.column_multiple = variant.get('column_multiple', 1)
            self.row_multiple = variant.get('row_multiple', 1)
            self.current_variant = variant
            self.variant_index = variant_index
            # Update max coordinates if position is already set
            if hasattr(self, 'x_min') and hasattr(self, 'y_min'):
                self.x_max = self.x_min + self.width
                self.y_max = self.y_min + self.height


class OptimizationUnit:
    """Represents a symmetry unit for optimization."""

    def __init__(self, l_half, r_half):
        self.l_half = l_half  # Left half block
        self.r_half = r_half  # Right half block
        self.x_child = None
        self.y_child = None
        self.parent = None


class OptimizationTreeNode:
    """Represents a B* tree node for optimization."""

    def __init__(self, name):
        self.name = name
        self.units = []
        self.x_child = None
        self.y_child = None
        self.parent = None
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.color = "FFFFFF"


class Timer:
    """Simple timer class for timeout handling."""

    def __init__(self, timeout_seconds):
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds

    def is_timeout(self):
        """Check if timeout has been reached."""
        return (time.time() - self.start_time) > self.timeout_seconds


class SimulatedAnnealingOptimizer:
    """Main class for optimizing block placement using Simulated Annealing."""

    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.blocks = {}  # name -> OptimizationBlock
        self.tree_root = None
        self.nets = []  # List of (net_name, [blocks])
        self.symmetry_constraints = {}  # Store symmetry information

        # Optimization parameters
        self.initial_temp = 2000.0
        self.final_temp = 1.0
        self.cooling_rate = 0.98
        self.max_iterations = 2000
        self.timeout_seconds = 120
        self.patience = 500
        self.aspect_ratio_target = 2.0

        # Cost function weights
        self.weight_area = 1.0
        self.weight_utilization = 10.0
        self.weight_wirelength = 100.0
        self.weight_aspect_ratio = 50.0

        # Random number generator
        self.rng = random.Random()

        # Statistics
        self.placement_statistics = {}

    def set_optimization_parameters(self, initial_temp=None, final_temp=None,
                                    cooling_rate=None, max_iterations=None,
                                    timeout_seconds=None, patience=None,
                                    aspect_ratio_target=None):
        """Set optimization parameters."""
        if initial_temp is not None:
            self.initial_temp = initial_temp
        if final_temp is not None:
            self.final_temp = final_temp
        if cooling_rate is not None:
            self.cooling_rate = cooling_rate
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        if patience is not None:
            self.patience = patience
        if aspect_ratio_target is not None:
            self.aspect_ratio_target = aspect_ratio_target

    def load_json_file(self, input_filename):
        """Loads a JSON file and stores the data."""
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                self.original_data = json.load(f)
            self.processed_data = copy.deepcopy(self.original_data)
            return self.original_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            self.original_data = None
            self.processed_data = None
            return None

    def load_from_n8n(self):
        """Loads JSON data from stdin (for n8n integration)."""
        try:
            self.original_data = json.load(sys.stdin)
            self.processed_data = copy.deepcopy(self.original_data)
            return self.original_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from stdin: {e}", file=sys.stderr)
            self.original_data = None
            self.processed_data = None
            return None

    def extract_optimization_data(self):
        """Extract blocks, tree structure, and nets from JSON data."""
        if not self.original_data or 'bstar_tree' not in self.original_data:
            print("No B* tree data found in JSON.")
            return False

        # First extract blocks from original blocks section to get variants
        if 'blocks' in self.original_data:
            for block_data in self.original_data['blocks']:
                block = OptimizationBlock(block_data['name'])
                block.device_type = block_data.get('device_type', '')
                block.variants = block_data.get('variants', [])

                # Set initial variant (default variant)
                if block.variants:
                    default_variant = None
                    variant_index = 0
                    for i, variant in enumerate(block.variants):
                        if variant.get('is_default', False):
                            default_variant = variant
                            variant_index = i
                            break
                    if not default_variant:
                        default_variant = block.variants[0]
                        variant_index = 0
                    block.set_variant(default_variant, variant_index)

                # Store symmetry constraints
                if 'symmetry' in block_data:
                    symmetry = block_data['symmetry']
                    self.symmetry_constraints[block.name] = symmetry

                self.blocks[block.name] = block

        # Parse B* tree structure and extract current positions/variants from bstar_tree
        bstar_tree_data = self.original_data['bstar_tree']
        if 'root' in bstar_tree_data:
            self.tree_root = self.parse_tree_node(bstar_tree_data['root'])

        # Extract nets information for wirelength calculation
        if 'netlist' in self.original_data and 'nets' in self.original_data['netlist']:
            self.extract_nets()

        return True

    def parse_tree_node(self, node_data):
        """Recursively parse B* tree node from JSON data."""
        if not node_data:
            return None

        node = OptimizationTreeNode(node_data['name'])
        node.x_min = node_data.get('x_min', 0.0)
        node.y_min = node_data.get('y_min', 0.0)
        node.x_max = node_data.get('x_max', 0.0)
        node.y_max = node_data.get('y_max', 0.0)
        node.color = node_data.get('color', 'FFFFFF')

        # Parse units and extract block information
        if 'units' in node_data:
            for unit_data in node_data['units']:
                l_half_data = unit_data.get('l_half', {})
                r_half_data = unit_data.get('r_half', {})

                # Update blocks with current positions and variants
                l_half_name = l_half_data.get('name', '')
                r_half_name = r_half_data.get('name', '')

                if l_half_name in self.blocks and r_half_name in self.blocks:
                    l_block = self.blocks[l_half_name]
                    r_block = self.blocks[r_half_name]

                    # Update l_half block
                    self.update_block_from_data(l_block, l_half_data)
                    # Update r_half block  
                    self.update_block_from_data(r_block, r_half_data)

                    # Create optimization unit
                    unit = OptimizationUnit(l_block, r_block)
                    node.units.append(unit)

        # Parse child nodes
        node.x_child = self.parse_tree_node(node_data.get('x_child'))
        node.y_child = self.parse_tree_node(node_data.get('y_child'))

        return node

    def update_block_from_data(self, block, block_data):
        """Update block with data from JSON."""
        block.width = block_data.get('width', 0.0)
        block.height = block_data.get('height', 0.0)
        block.x_min = block_data.get('x_min', 0.0)
        block.y_min = block_data.get('y_min', 0.0)
        block.x_max = block_data.get('x_max', 0.0)
        block.y_max = block_data.get('y_max', 0.0)
        block.color = block_data.get('color', 'FFFFFF')
        block.current_variant = block_data.get('current_variant')
        block.variant_index = block_data.get('variant_index', 0)

    def extract_nets(self):
        """Extract net connectivity information for wirelength calculation."""
        if 'netlist' not in self.original_data:
            return

        # Create mapping from instance name to block
        instance_to_block = {}
        if 'instances' in self.original_data['netlist']:
            for instance in self.original_data['netlist']['instances']:
                instance_name = instance['name']
                block_name = instance.get('block', instance_name)
                if block_name in self.blocks:
                    instance_to_block[instance_name] = self.blocks[block_name]

        # Build nets list
        self.nets = []
        for net_data in self.original_data['netlist']['nets']:
            net_name = net_data['name']
            connected_instances = net_data.get('connected_instances', [])

            # Map instances to blocks
            net_blocks = []
            for instance_name in connected_instances:
                if instance_name in instance_to_block:
                    block = instance_to_block[instance_name]
                    if block not in net_blocks:  # Avoid duplicates
                        net_blocks.append(block)

            if len(net_blocks) > 1:  # Only consider nets with multiple blocks
                self.nets.append((net_name, net_blocks))

    def calculate_cost(self):
        """Calculate the total cost of the current placement."""
        # Check for hard constraint violations
        if self.has_overlaps() or self.violates_symmetry():
            return float('inf')

        # Calculate individual cost components
        area_cost = self.calculate_area_cost()
        utilization_cost = self.calculate_utilization_cost()
        wirelength_cost = self.calculate_wirelength_cost()
        aspect_ratio_cost = self.calculate_aspect_ratio_cost()

        # Weighted sum of costs
        total_cost = (self.weight_area * area_cost +
                      self.weight_utilization * utilization_cost +
                      self.weight_wirelength * wirelength_cost +
                      self.weight_aspect_ratio * aspect_ratio_cost)

        return total_cost

    def has_overlaps(self):
        """Check if any blocks overlap."""
        block_list = list(self.blocks.values())
        overlaps = 0

        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                if self.blocks_overlap(block_list[i], block_list[j]):
                    overlaps += 1

        # Allow some minor overlaps (up to 2) as soft constraint
        return overlaps > 2

    def blocks_overlap(self, block1, block2):
        """Check if two blocks overlap."""
        tolerance = 0.01  # Small tolerance for floating point errors

        # Check if blocks are completely separate
        if (block1.x_max <= block2.x_min + tolerance or
                block2.x_max <= block1.x_min + tolerance or
                block1.y_max <= block2.y_min + tolerance or
                block2.y_max <= block1.y_min + tolerance):
            return False

        # Check if it's just a tiny overlap (touching edges)
        x_overlap = min(block1.x_max, block2.x_max) - max(block1.x_min, block2.x_min)
        y_overlap = min(block1.y_max, block2.y_max) - max(block1.y_min, block2.y_min)

        # Allow very small overlaps (essentially touching)
        return x_overlap > tolerance and y_overlap > tolerance

    def calculate_cost(self):
        """Calculate the total cost of the current placement."""
        # Check for serious violations only
        serious_overlaps = self.count_serious_overlaps()
        serious_symmetry_violations = self.count_serious_symmetry_violations()

        # Hard constraints with high but finite penalty
        if serious_overlaps > 0 or serious_symmetry_violations > 0:
            penalty = 1000000.0 * (serious_overlaps + serious_symmetry_violations)
            return penalty

        # Calculate individual cost components
        area_cost = self.calculate_area_cost()
        utilization_cost = self.calculate_utilization_cost()
        wirelength_cost = self.calculate_wirelength_cost()
        aspect_ratio_cost = self.calculate_aspect_ratio_cost()

        # Weighted sum of costs
        total_cost = (self.weight_area * area_cost +
                      self.weight_utilization * utilization_cost +
                      self.weight_wirelength * wirelength_cost +
                      self.weight_aspect_ratio * aspect_ratio_cost)

        return total_cost

    def count_serious_overlaps(self):
        """Count overlaps that are more than just touching."""
        block_list = list(self.blocks.values())
        serious_overlaps = 0

        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                if self.blocks_seriously_overlap(block_list[i], block_list[j]):
                    serious_overlaps += 1

        return serious_overlaps

    def blocks_seriously_overlap(self, block1, block2):
        """Check if blocks have significant overlap (not just touching)."""
        tolerance = 0.1  # Larger tolerance

        # Check if blocks are separate
        if (block1.x_max <= block2.x_min + tolerance or
                block2.x_max <= block1.x_min + tolerance or
                block1.y_max <= block2.y_min + tolerance or
                block2.y_max <= block1.y_min + tolerance):
            return False

        # Check overlap area
        x_overlap = min(block1.x_max, block2.x_max) - max(block1.x_min, block2.x_min)
        y_overlap = min(block1.y_max, block2.y_max) - max(block1.y_min, block2.y_min)

        if x_overlap <= tolerance or y_overlap <= tolerance:
            return False

        overlap_area = x_overlap * y_overlap
        min_block_area = min(block1.width * block1.height, block2.width * block2.height)

        # Only consider it serious if overlap is > 5% of smaller block
        return overlap_area > 0.05 * min_block_area

    def count_serious_symmetry_violations(self):
        """Count serious symmetry constraint violations."""
        violations = 0

        for block_name, constraint in self.symmetry_constraints.items():
            if constraint.get('type') == 'pair_symmetric':
                pair_with = constraint.get('pair_with')
                if pair_with and pair_with in self.blocks:
                    block1 = self.blocks[block_name]
                    block2 = self.blocks[pair_with]

                    # Check if they are very far apart
                    distance = self.calculate_block_distance(block1, block2)
                    max_dimension = max(block1.width, block1.height, block2.width, block2.height)

                    # Allow distance up to 3x the largest dimension
                    if distance > 3.0 * max_dimension:
                        violations += 1

        return violations

    def are_blocks_adjacent(self, block1, block2):
        """Check if two blocks are adjacent (touching sides)."""
        # Horizontal adjacency
        if (abs(block1.x_max - block2.x_min) < 1e-6 or
                abs(block2.x_max - block1.x_min) < 1e-6):
            # Check if they overlap in y direction
            return not (block1.y_max <= block2.y_min or block2.y_max <= block1.y_min)

        # Vertical adjacency  
        if (abs(block1.y_max - block2.y_min) < 1e-6 or
                abs(block2.y_max - block1.y_min) < 1e-6):
            # Check if they overlap in x direction
            return not (block1.x_max <= block2.x_min or block2.x_max <= block1.x_min)

        return False

    def calculate_area_cost(self):
        """Calculate cost based on total bounding box area."""
        if not self.blocks:
            return 0.0

        blocks_list = list(self.blocks.values())
        x_min = min(block.x_min for block in blocks_list)
        x_max = max(block.x_max for block in blocks_list)
        y_min = min(block.y_min for block in blocks_list)
        y_max = max(block.y_max for block in blocks_list)

        return (x_max - x_min) * (y_max - y_min)

    def calculate_utilization_cost(self):
        """Calculate cost based on area utilization (lower utilization = higher cost)."""
        if not self.blocks:
            return 0.0

        # Total block area
        total_block_area = sum(block.width * block.height for block in self.blocks.values())

        # Bounding box area
        bounding_box_area = self.calculate_area_cost()

        if bounding_box_area == 0:
            return float('inf')

        utilization = total_block_area / bounding_box_area

        # Return inverse of utilization as cost (higher utilization = lower cost)
        return 100.0 / max(utilization, 0.01)  # Avoid division by zero

    def calculate_wirelength_cost(self):
        """Calculate total Half-Perimeter Wire Length (HPWL)."""
        total_hpwl = 0.0

        for net_name, net_blocks in self.nets:
            if len(net_blocks) < 2:
                continue

            # Find bounding box of all blocks in this net
            x_coords = [block.get_center_x() for block in net_blocks]
            y_coords = [block.get_center_y() for block in net_blocks]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            # HPWL = half perimeter of bounding box
            hpwl = (x_max - x_min) + (y_max - y_min)
            total_hpwl += hpwl

        return total_hpwl

    def calculate_aspect_ratio_cost(self):
        """Calculate cost based on deviation from target aspect ratio."""
        if not self.blocks:
            return 0.0

        blocks_list = list(self.blocks.values())
        x_min = min(block.x_min for block in blocks_list)
        x_max = max(block.x_max for block in blocks_list)
        y_min = min(block.y_min for block in blocks_list)
        y_max = max(block.y_max for block in blocks_list)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0 or height == 0:
            return float('inf')

        aspect_ratio = height / width
        deviation = abs(aspect_ratio - self.aspect_ratio_target)

        return deviation * deviation  # Quadratic penalty

    def apply_perturbation(self):
        """Apply a random perturbation to the current placement."""
        perturbation_types = [
            self.swap_tree_children,
            self.change_block_variant,
            self.move_subtree,
            self.flip_symmetry_unit
        ]

        # Apply one perturbation
        perturbation = self.rng.choice(perturbation_types)
        try:
            perturbation()
        except Exception as e:
            print(f"Warning: Perturbation failed: {e}")
            # If perturbation fails, just continue
            pass

    def swap_tree_children(self):
        """Swap x_child and y_child of a random tree node."""
        if not self.tree_root:
            return

        # Collect all nodes with at least one child
        nodes_with_children = []
        self.collect_nodes_with_children(self.tree_root, nodes_with_children)

        if nodes_with_children:
            node = self.rng.choice(nodes_with_children)
            # Simple swap of children
            node.x_child, node.y_child = node.y_child, node.x_child

    def collect_nodes_with_children(self, node, nodes_list):
        """Recursively collect nodes that have children."""
        if not node:
            return

        if node.x_child or node.y_child:
            nodes_list.append(node)

        self.collect_nodes_with_children(node.x_child, nodes_list)
        self.collect_nodes_with_children(node.y_child, nodes_list)

    def change_block_variant(self):
        """Change the variant of a random block."""
        blocks_with_variants = [block for block in self.blocks.values()
                                if len(block.variants) > 1]

        if blocks_with_variants:
            block = self.rng.choice(blocks_with_variants)
            # Choose a different variant
            current_index = block.variant_index
            available_indices = [i for i in range(len(block.variants)) if i != current_index]

            if available_indices:
                new_variant_index = self.rng.choice(available_indices)
                new_variant = block.variants[new_variant_index]
                block.set_variant(new_variant, new_variant_index)

    def move_subtree(self):
        """Move a subtree to a different position in the tree."""
        if not self.tree_root:
            return

        # Collect all nodes (potential subtree roots)
        all_nodes = []
        self.collect_all_nodes(self.tree_root, all_nodes)

        if len(all_nodes) < 3:  # Need at least root + 2 others to move
            return

        # Choose a subtree to move (not the root)
        moveable_nodes = [node for node in all_nodes if node != self.tree_root]
        if not moveable_nodes:
            return

        subtree_node = self.rng.choice(moveable_nodes)

        # Find its current parent and remove it
        parent = self.find_parent(self.tree_root, subtree_node)
        if not parent:
            return

        # Disconnect subtree from current parent
        if parent.x_child == subtree_node:
            parent.x_child = None
        elif parent.y_child == subtree_node:
            parent.y_child = None

        # Find a new parent (that has a free slot)
        potential_parents = [node for node in all_nodes
                             if node != subtree_node and (not node.x_child or not node.y_child)]

        if potential_parents:
            new_parent = self.rng.choice(potential_parents)

            # Attach to new parent
            if not new_parent.x_child:
                new_parent.x_child = subtree_node
            elif not new_parent.y_child:
                new_parent.y_child = subtree_node

    def find_parent(self, root, target_node):
        """Find the parent of target_node in the tree rooted at root."""
        if not root or root == target_node:
            return None

        if root.x_child == target_node or root.y_child == target_node:
            return root

        # Search in children
        result = self.find_parent(root.x_child, target_node)
        if result:
            return result

        return self.find_parent(root.y_child, target_node)

    def collect_all_nodes(self, node, nodes_list):
        """Recursively collect all tree nodes."""
        if not node:
            return

        nodes_list.append(node)
        self.collect_all_nodes(node.x_child, nodes_list)
        self.collect_all_nodes(node.y_child, nodes_list)

    def flip_symmetry_unit(self):
        """Flip the order in a symmetry unit (swap l_half and r_half)."""
        all_units = []
        self.collect_all_units(self.tree_root, all_units)

        # Only flip units where l_half != r_half (actual symmetric pairs)
        flippable_units = [unit for unit in all_units if unit.l_half != unit.r_half]

        if flippable_units:
            unit = self.rng.choice(flippable_units)
            unit.l_half, unit.r_half = unit.r_half, unit.l_half

    def collect_all_units(self, node, units_list):
        """Recursively collect all symmetry units."""
        if not node:
            return

        units_list.extend(node.units)
        self.collect_all_units(node.x_child, units_list)
        self.collect_all_units(node.y_child, units_list)

    def place_blocks(self):
        """Place blocks according to current B* tree structure."""
        if not self.tree_root:
            print("Warning: No tree root for placement")
            return

        # Reset all block positions
        for block in self.blocks.values():
            block.x_min = 0.0
            block.y_min = 0.0
            block.x_max = 0.0
            block.y_max = 0.0

        try:
            # Place blocks starting from root
            self.place_tree_node(self.tree_root, 0.0, 0.0)
        except Exception as e:
            print(f"Warning: Placement failed: {e}")
            # If placement fails, set a default layout
            self.set_default_placement()

    def set_default_placement(self):
        """Set a simple default placement if tree placement fails."""
        current_x = 0.0
        current_y = 0.0
        max_height = 0.0

        for block in self.blocks.values():
            block.set_position(current_x, current_y)
            current_x += block.width
            max_height = max(max_height, block.height)

            # Start new row if getting too wide
            if current_x > 50.0:
                current_x = 0.0
                current_y += max_height
                max_height = 0.0

    def place_tree_node(self, node, x_offset=0.0, y_offset=0.0):
        """Place a tree node and its children recursively."""
        if not node:
            return {'x_min': x_offset, 'y_min': y_offset, 'x_max': x_offset, 'y_max': y_offset}

        placed_blocks = []

        # Place units within this node
        current_y = y_offset
        max_width = 0.0

        for unit in node.units:
            if not unit.l_half or not unit.r_half:
                continue

            if unit.r_half.name == unit.l_half.name:
                # Single block (self-symmetric or individual)
                block = unit.r_half
                block.set_position(x_offset, current_y)
                placed_blocks.append(block)
                current_y += block.height
                max_width = max(max_width, block.width)
            else:
                # Symmetric pair - place side by side
                l_block = unit.l_half
                r_block = unit.r_half

                # Place left block
                l_block.set_position(x_offset, current_y)
                # Place right block next to left block
                r_block.set_position(x_offset + l_block.width, current_y)

                placed_blocks.extend([l_block, r_block])
                current_y += max(l_block.height, r_block.height)
                max_width = max(max_width, l_block.width + r_block.width)

        # Calculate bounding box for this node
        if placed_blocks:
            node.x_min = min(block.x_min for block in placed_blocks)
            node.y_min = min(block.y_min for block in placed_blocks)
            node.x_max = max(block.x_max for block in placed_blocks)
            node.y_max = max(block.y_max for block in placed_blocks)
        else:
            node.x_min = x_offset
            node.y_min = y_offset
            node.x_max = x_offset + max_width
            node.y_max = current_y

        # Place x_child (horizontally adjacent)
        if node.x_child:
            x_child_bbox = self.place_tree_node(node.x_child, node.x_max, node.y_min)
            node.x_max = max(node.x_max, x_child_bbox['x_max'])
            node.y_max = max(node.y_max, x_child_bbox['y_max'])

        # Place y_child (vertically adjacent)
        if node.y_child:
            y_child_bbox = self.place_tree_node(node.y_child, node.x_min, node.y_max)
            node.x_max = max(node.x_max, y_child_bbox['x_max'])
            node.y_max = max(node.y_max, y_child_bbox['y_max'])

        return {
            'x_min': node.x_min,
            'y_min': node.y_min,
            'x_max': node.x_max,
            'y_max': node.y_max
        }

    def save_current_state(self):
        """Save the current state of blocks and tree."""
        # Create a full deep copy of the current state
        state = {
            'blocks': {},
            'tree_root': copy.deepcopy(self.tree_root)
        }

        # Save block states (positions and variants)
        for name, block in self.blocks.items():
            state['blocks'][name] = {
                'x_min': block.x_min,
                'y_min': block.y_min,
                'x_max': block.x_max,
                'y_max': block.y_max,
                'width': block.width,
                'height': block.height,
                'variant_index': block.variant_index,
                'current_variant': copy.deepcopy(block.current_variant) if block.current_variant else None,
                'color': block.color
            }

        return state

    def restore_state(self, state):
        """Restore a previously saved state."""
        # Restore block states
        for name, data in state['blocks'].items():
            if name in self.blocks:
                block = self.blocks[name]
                block.x_min = data['x_min']
                block.y_min = data['y_min']
                block.x_max = data['x_max']
                block.y_max = data['y_max']
                block.width = data['width']
                block.height = data['height']
                block.variant_index = data['variant_index']
                block.current_variant = data['current_variant']
                block.color = data['color']

        # Restore tree structure
        self.tree_root = state['tree_root']

        # Rebuild unit references in tree to point to current blocks
        self.rebuild_tree_references(self.tree_root)

    def rebuild_tree_references(self, node):
        """Rebuild block references in tree units after state restoration."""
        if not node:
            return

        # Rebuild unit references
        for unit in node.units:
            if hasattr(unit.l_half, 'name'):
                unit.l_half = self.blocks[unit.l_half.name]
            if hasattr(unit.r_half, 'name'):
                unit.r_half = self.blocks[unit.r_half.name]

        # Recursively rebuild children
        self.rebuild_tree_references(node.x_child)
        self.rebuild_tree_references(node.y_child)

    def violates_symmetry(self):
        """Check if symmetry constraints are violated."""
        for block_name, constraint in self.symmetry_constraints.items():
            if constraint.get('type') == 'pair_symmetric':
                pair_with = constraint.get('pair_with')
                if pair_with and pair_with in self.blocks:
                    block1 = self.blocks[block_name]
                    block2 = self.blocks[pair_with]

                    # For now, just check if they are reasonably close
                    # (More lenient than strict adjacency)
                    distance = self.calculate_block_distance(block1, block2)
                    if distance > 5.0:  # Allow some tolerance
                        return True
        return False

    def calculate_block_distance(self, block1, block2):
        """Calculate distance between centers of two blocks."""
        center1_x = block1.get_center_x()
        center1_y = block1.get_center_y()
        center2_x = block2.get_center_x()
        center2_y = block2.get_center_y()

        return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    def are_blocks_adjacent(self, block1, block2):
        """Check if two blocks are adjacent (touching sides)."""
        tolerance = 0.01  # Small tolerance for floating point errors

        # Horizontal adjacency
        if (abs(block1.x_max - block2.x_min) < tolerance or
                abs(block2.x_max - block1.x_min) < tolerance):
            # Check if they overlap in y direction
            return not (block1.y_max <= block2.y_min + tolerance or
                        block2.y_max <= block1.y_min + tolerance)

        # Vertical adjacency  
        if (abs(block1.y_max - block2.y_min) < tolerance or
                abs(block2.y_max - block1.y_min) < tolerance):
            # Check if they overlap in x direction
            return not (block1.x_max <= block2.x_min + tolerance or
                        block2.x_max <= block1.x_min + tolerance)

        return False

    def run_simulated_annealing(self, use_fixed_seed=True):
        """Run the simulated annealing optimization."""
        if use_fixed_seed:
            self.rng.seed(42)  # Fixed seed for reproducibility
        else:
            self.rng.seed()  # Random seed

        print("Starting Simulated Annealing optimization...")
        print(f"Parameters: T_initial={self.initial_temp}, T_final={self.final_temp}")
        print(f"Cooling_rate={self.cooling_rate}, Max_iter={self.max_iterations}")
        print(f"Timeout={self.timeout_seconds}s, Patience={self.patience}")
        print(f"Aspect_ratio_target={self.aspect_ratio_target}")

        # Initialize placement
        self.place_blocks()

        # Initial state and cost
        best_cost = self.calculate_cost()
        current_cost = best_cost
        best_state = self.save_current_state()

        print(f"Initial cost: {current_cost:.6f}")

        # Check if initial placement is valid
        if current_cost >= 1000000.0:
            print("Warning: Initial placement has constraint violations!")

        # SA parameters
        temperature = self.initial_temp
        iteration = 0
        accepts = 0
        rejects = 0
        consecutive_rejects = 0

        timer = Timer(self.timeout_seconds)

        # Main SA loop
        while (iteration < self.max_iterations and
               temperature > self.final_temp and
               consecutive_rejects < self.patience and
               not timer.is_timeout()):

            # Save current state
            current_state = self.save_current_state()

            # Apply perturbation
            self.apply_perturbation()

            # Update placement
            self.place_blocks()

            # Calculate new cost
            new_cost = self.calculate_cost()

            # Accept or reject move
            if new_cost < current_cost:
                # Accept improvement
                current_cost = new_cost
                accepts += 1
                consecutive_rejects = 0

                # Update best solution
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = self.save_current_state()
                    print(f"Iteration {iteration}: New best cost = {best_cost:.6f}")
            else:
                # Metropolis criterion
                delta = new_cost - current_cost
                probability = math.exp(-delta / temperature) if temperature > 0 else 0

                if self.rng.random() < probability:
                    # Accept worse solution
                    current_cost = new_cost
                    accepts += 1
                    consecutive_rejects = 0
                else:
                    # Reject move
                    self.restore_state(current_state)
                    rejects += 1
                    consecutive_rejects += 1

            # Cool down
            temperature *= self.cooling_rate
            iteration += 1

            # Progress reporting
            if iteration % 100 == 0:
                accept_rate = accepts / (accepts + rejects) * 100 if (accepts + rejects) > 0 else 0
                print(f"Iteration {iteration}: T={temperature:.2f}, "
                      f"Current={current_cost:.6f}, Best={best_cost:.6f}, "
                      f"Accept_rate={accept_rate:.1f}%")

                # Debug info for low acceptance rates
                if accept_rate < 5.0 and iteration > 200:
                    print(f"  Debug: Checking constraint violations...")
                    overlaps = self.count_serious_overlaps()
                    sym_violations = self.count_serious_symmetry_violations()
                    print(f"  Serious overlaps: {overlaps}, Symmetry violations: {sym_violations}")

        # Restore best solution
        self.restore_state(best_state)
        self.place_blocks()

        # Final statistics
        print(f"\nOptimization completed:")
        print(f"Final cost: {best_cost:.6f}")
        print(f"Total iterations: {iteration}")
        print(f"Accepts: {accepts}, Rejects: {rejects}")
        print(f"Final temperature: {temperature:.6f}")

        if timer.is_timeout():
            print("Stopped due to timeout")
        elif consecutive_rejects >= self.patience:
            print("Stopped due to patience limit")
        elif iteration >= self.max_iterations:
            print("Stopped due to iteration limit")
        elif temperature <= self.final_temp:
            print("Stopped due to temperature limit")

        return best_cost

    def calculate_placement_statistics(self):
        """Calculate statistics about the current placement."""
        placed_blocks = list(self.blocks.values())

        if not placed_blocks:
            self.placement_statistics = {
                "total_blocks": 0,
                "placed_blocks": 0,
                "total_block_area": 0.0,
                "bounding_box": {"width": 0.0, "height": 0.0, "area": 0.0},
                "utilization_percent": 0.0,
                "aspect_ratio": 0.0
            }
            return

        # Calculate bounding box
        x_min = min(block.x_min for block in placed_blocks)
        y_min = min(block.y_min for block in placed_blocks)
        x_max = max(block.x_max for block in placed_blocks)
        y_max = max(block.y_max for block in placed_blocks)

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area = bbox_width * bbox_height

        # Calculate total block area
        total_block_area = sum(block.width * block.height for block in placed_blocks)

        # Calculate utilization
        utilization = (total_block_area / bbox_area * 100) if bbox_area > 0 else 0.0

        # Count blocks by device type
        device_type_counts = {}
        for block in placed_blocks:
            device_type = block.device_type or "unknown"
            device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1

        self.placement_statistics = {
            "total_blocks": len(self.blocks),
            "placed_blocks": len(placed_blocks),
            "total_block_area": round(total_block_area, 6),
            "bounding_box": {
                "x_min": round(x_min, 6),
                "y_min": round(y_min, 6),
                "x_max": round(x_max, 6),
                "y_max": round(y_max, 6),
                "width": round(bbox_width, 6),
                "height": round(bbox_height, 6),
                "area": round(bbox_area, 6)
            },
            "utilization_percent": round(utilization, 2),
            "device_type_counts": device_type_counts,
            "aspect_ratio": round(bbox_height / bbox_width, 3) if bbox_width > 0 else 0.0
        }

    def update_tree_coordinates_in_json(self, node_data, placement_node):
        """Update coordinates in the JSON tree structure."""
        if not node_data or not placement_node:
            return

        # Update node coordinates and color
        node_data['x_min'] = round(placement_node.x_min, 6)
        node_data['y_min'] = round(placement_node.y_min, 6)
        node_data['x_max'] = round(placement_node.x_max, 6)
        node_data['y_max'] = round(placement_node.y_max, 6)
        node_data['color'] = placement_node.color

        # Update unit coordinates
        if 'units' in node_data and placement_node.units:
            for i, unit_data in enumerate(node_data['units']):
                if i < len(placement_node.units):
                    unit = placement_node.units[i]

                    # Update l_half coordinates
                    if 'l_half' in unit_data and unit.l_half:
                        l_half_data = unit_data['l_half']
                        l_half_data['x_min'] = round(unit.l_half.x_min, 6)
                        l_half_data['y_min'] = round(unit.l_half.y_min, 6)
                        l_half_data['x_max'] = round(unit.l_half.x_max, 6)
                        l_half_data['y_max'] = round(unit.l_half.y_max, 6)
                        l_half_data['width'] = round(unit.l_half.width, 6)
                        l_half_data['height'] = round(unit.l_half.height, 6)
                        l_half_data['color'] = unit.l_half.color
                        if unit.l_half.current_variant:
                            l_half_data['current_variant'] = unit.l_half.current_variant
                            l_half_data['variant_index'] = unit.l_half.variant_index

                    # Update r_half coordinates
                    if 'r_half' in unit_data and unit.r_half:
                        r_half_data = unit_data['r_half']
                        r_half_data['x_min'] = round(unit.r_half.x_min, 6)
                        r_half_data['y_min'] = round(unit.r_half.y_min, 6)
                        r_half_data['x_max'] = round(unit.r_half.x_max, 6)
                        r_half_data['y_max'] = round(unit.r_half.y_max, 6)
                        r_half_data['width'] = round(unit.r_half.width, 6)
                        r_half_data['height'] = round(unit.r_half.height, 6)
                        r_half_data['color'] = unit.r_half.color
                        if unit.r_half.current_variant:
                            r_half_data['current_variant'] = unit.r_half.current_variant
                            r_half_data['variant_index'] = unit.r_half.variant_index

        # Recursively update children
        if 'x_child' in node_data and node_data['x_child'] and placement_node.x_child:
            self.update_tree_coordinates_in_json(node_data['x_child'], placement_node.x_child)

        if 'y_child' in node_data and node_data['y_child'] and placement_node.y_child:
            self.update_tree_coordinates_in_json(node_data['y_child'], placement_node.y_child)

    def update_processed_data(self):
        """Update the processed data with optimized placement information."""
        if not self.processed_data or 'bstar_tree' not in self.processed_data:
            return False

        # Update tree coordinates
        if 'root' in self.processed_data['bstar_tree'] and self.tree_root:
            self.update_tree_coordinates_in_json(
                self.processed_data['bstar_tree']['root'],
                self.tree_root
            )

        # Calculate and update placement statistics
        self.calculate_placement_statistics()
        self.processed_data['bstar_tree']['placement_statistics'] = self.placement_statistics

        # Add optimization information
        self.processed_data['bstar_tree']['optimization_info'] = {
            "optimization_method": "simulated_annealing",
            "optimization_completed": True,
            "final_cost": round(self.calculate_cost(), 6),
            "final_utilization": self.placement_statistics.get('utilization_percent', 0.0),
            "final_aspect_ratio": self.placement_statistics.get('aspect_ratio', 0.0)
        }

        # Update description
        self.processed_data['bstar_tree']['description'] = "B* tree structure with optimized block placement"

        return True

    def save_to_file(self, output_filename):
        """Save processed data to JSON file."""
        if not self.processed_data:
            print("No processed data to save.")
            return False

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return False

    def output_to_n8n(self):
        """Output processed data to stdout (for n8n integration)."""
        if self.processed_data is not None:
            print(json.dumps(self.processed_data))
        else:
            print("No processed data available.", file=sys.stderr)

    def process_optimization(self, use_fixed_seed=True):
        """Main processing method that performs optimization."""
        if not self.extract_optimization_data():
            return False

        if not self.run_simulated_annealing(use_fixed_seed):
            return False

        if not self.update_processed_data():
            return False

        return True

    def main_n8n(self):
        """Main method for n8n integration."""
        if not self.load_from_n8n():
            return False

        if not self.process_optimization(use_fixed_seed=False):  # Use random seed for n8n
            return False

        self.output_to_n8n()
        return True

    def main_local(self, input_filename, output_filename, use_fixed_seed=True):
        """Main method for local file processing."""
        input_file = f"{input_filename}.json"
        output_file = f"{output_filename}.json"

        if not self.load_json_file(input_file):
            print(f"Error: Could not load input file {input_file}")
            return False

        if not self.process_optimization(use_fixed_seed):
            print(f"Error: Could not perform optimization")
            return False

        if not self.save_to_file(output_file):
            print(f"Error: Could not save output file {output_file}")
            return False

        print(f"Successfully optimized placement:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Optimized {self.placement_statistics['placed_blocks']} blocks")
        print(
            f"  Final bounding box: {self.placement_statistics['bounding_box']['width']:.2f} x {self.placement_statistics['bounding_box']['height']:.2f}")
        print(f"  Final utilization: {self.placement_statistics['utilization_percent']:.2f}%")
        print(f"  Final aspect ratio: {self.placement_statistics['aspect_ratio']:.3f}")
        print(f"  Final cost: {self.calculate_cost():.6f}")

        return True


def optimize_placement_from_files(input_filename, output_filename, use_fixed_seed=True):
    """
    Convenience function to optimize placement from input file and save to output file.

    Args:
        input_filename (str): Input JSON filename (without extension)
        output_filename (str): Output JSON filename (without extension)
        use_fixed_seed (bool): Whether to use fixed seed for reproducibility

    Returns:
        bool: True if successful, False otherwise
    """
    optimizer = SimulatedAnnealingOptimizer()

    # Set default optimization parameters (can be modified before calling)
    optimizer.set_optimization_parameters(
        initial_temp=2000.0,
        final_temp=1.0,
        cooling_rate=0.98,
        max_iterations=2000,
        timeout_seconds=120,
        patience=500,
        aspect_ratio_target=2.0
    )

    success = optimizer.main_local(input_filename, output_filename, use_fixed_seed)

    if success:
        print(f"Successfully completed placement optimization.")
    else:
        print(f"Failed to complete placement optimization.")

    return success


if __name__ == "__main__":
    # n8n integration mode only
    optimizer = SimulatedAnnealingOptimizer()
    optimizer.main_n8n()