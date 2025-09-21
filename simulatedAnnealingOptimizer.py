import json
import sys
import random
import copy
import math
import time
from typing import Dict, List, Tuple, Optional, Any


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
        self.column_multiple = 1
        self.row_multiple = 1
        self.variant_index = 0
        self.current_variant = None
        self.all_variants = []
        self.device_type = ""
        self.color = "FFFFFF"
        self.is_placed = False

    def get_width(self):
        return self.x_max - self.x_min if self.x_max > self.x_min else self.width

    def get_height(self):
        return self.y_max - self.y_min if self.y_max > self.y_min else self.height

    def get_center_x(self):
        return (self.x_min + self.x_max) * 0.5

    def get_center_y(self):
        return (self.y_min + self.y_max) * 0.5

    def set_position(self, x_min, y_min):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_min + self.width
        self.y_max = y_min + self.height
        self.is_placed = True

    def set_variant(self, variant, variant_index=0):
        """Set the variant for this block."""
        if variant:
            self.width = variant['width']
            self.height = variant['height']
            self.column_multiple = variant.get('column_multiple', 1)
            self.row_multiple = variant.get('row_multiple', 1)
            self.current_variant = variant
            self.variant_index = variant_index
            # Update x_max and y_max if position is already set
            if self.is_placed:
                self.x_max = self.x_min + self.width
                self.y_max = self.y_min + self.height


class OptimizationUnit:
    """Represents a symmetry unit for optimization."""

    def __init__(self, l_half, r_half):
        self.l_half = l_half
        self.r_half = r_half
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
    """Simple timer class for optimization timeout."""

    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def is_timeout(self):
        return (time.time() - self.start_time) > self.timeout_seconds


class CostFunction:
    """Cost function for evaluating placement quality."""

    def __init__(self, aspect_ratio_target=2.0, weight_area=1.0, weight_wirelength=1.0, weight_aspect=0.5):
        self.aspect_ratio_target = aspect_ratio_target
        self.weight_area = weight_area
        self.weight_wirelength = weight_wirelength
        self.weight_aspect = weight_aspect

    def __call__(self, optimizer):
        """Calculate total cost of current placement."""
        area_cost = self.calculate_area_cost(optimizer)
        wirelength_cost = self.calculate_wirelength_cost(optimizer)
        aspect_cost = self.calculate_aspect_ratio_cost(optimizer)

        total_cost = (self.weight_area * area_cost +
                      self.weight_wirelength * wirelength_cost +
                      self.weight_aspect * aspect_cost)

        return total_cost

    def calculate_area_cost(self, optimizer):
        """Calculate area-based cost."""
        if not optimizer.placement_statistics:
            return float('inf')

        bbox = optimizer.placement_statistics.get('bounding_box', {})
        total_area = bbox.get('area', 0.0)
        utilization = optimizer.placement_statistics.get('utilization_percent', 0.0)

        # Penalize large areas and low utilization
        if utilization == 0:
            return float('inf')

        return total_area / max(utilization, 1.0)

    def calculate_wirelength_cost(self, optimizer):
        """Calculate wirelength-based cost using HPWL."""
        if not optimizer.nets:
            return 0.0

        total_hpwl = 0.0
        for net_name, net_blocks in optimizer.nets:
            if len(net_blocks) < 2:
                continue

            x_coords = [block.get_center_x() for block in net_blocks]
            y_coords = [block.get_center_y() for block in net_blocks]

            if x_coords and y_coords:
                hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
                total_hpwl += hpwl

        return total_hpwl

    def calculate_aspect_ratio_cost(self, optimizer):
        """Calculate aspect ratio cost."""
        if not optimizer.placement_statistics:
            return float('inf')

        current_aspect = optimizer.placement_statistics.get('aspect_ratio', 1.0)
        aspect_deviation = abs(current_aspect - self.aspect_ratio_target)

        return aspect_deviation * 1000  # Scale the penalty


class Perturbator:
    """Handles different types of perturbations for optimization."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.rng = random.Random()

        # Available perturbation methods
        self.perturbation_methods = [
            self.random_swap_tree_children,
            self.random_resize_block_variant,
            self.random_flip_symmetry_unit,
            self.random_rebuild_tree_structure,
            self.random_transplant_subtree
        ]

    def __call__(self):
        """Apply random perturbations."""
        num_perturbations = self.rng.randint(1, 3)

        for _ in range(num_perturbations):
            if self.perturbation_methods:
                method = self.rng.choice(self.perturbation_methods)
                method()

    def random_swap_tree_children(self):
        """Randomly swap x_child and y_child of a tree node."""
        if not self.optimizer.tree_nodes:
            return

        node = self.rng.choice(self.optimizer.tree_nodes)
        node.x_child, node.y_child = node.y_child, node.x_child

    def random_resize_block_variant(self):
        """Randomly change the variant of a block."""
        if not self.optimizer.blocks:
            return

        # Find blocks with multiple variants
        resizable_blocks = [block for block in self.optimizer.blocks.values()
                            if len(block.all_variants) > 1]

        if not resizable_blocks:
            return

        block = self.rng.choice(resizable_blocks)
        new_variant_index = self.rng.randint(0, len(block.all_variants) - 1)
        new_variant = block.all_variants[new_variant_index]

        # Apply variant to symmetric blocks if needed
        symmetric_blocks = self.optimizer.get_symmetric_blocks(block)
        for sym_block in symmetric_blocks:
            sym_block.set_variant(new_variant, new_variant_index)

    def random_flip_symmetry_unit(self):
        """Randomly flip a symmetry unit."""
        if not self.optimizer.symmetry_units:
            return

        unit = self.rng.choice(self.optimizer.symmetry_units)
        if unit.l_half != unit.r_half:  # Only flip asymmetric units
            unit.x_child, unit.y_child = unit.y_child, unit.x_child

    def random_rebuild_tree_structure(self):
        """Randomly rebuild part of the tree structure."""
        if not self.optimizer.tree_nodes or len(self.optimizer.tree_nodes) < 2:
            return

        # Select a subtree to rebuild
        node = self.rng.choice(self.optimizer.tree_nodes)
        if len(node.units) > 1:
            self.rng.shuffle(node.units)
            # Rebuild the y_child chain for units
            for i in range(len(node.units) - 1):
                node.units[i].x_child = None
                node.units[i].y_child = node.units[i + 1]
                node.units[i + 1].parent = node.units[i]

            node.units[-1].x_child = None
            node.units[-1].y_child = None

    def random_transplant_subtree(self):
        """Randomly transplant a subtree to a different location."""
        if len(self.optimizer.tree_nodes) < 3:
            return

        # Select source and target nodes
        source = self.rng.choice(self.optimizer.tree_nodes)
        target = self.rng.choice([n for n in self.optimizer.tree_nodes if n != source])

        # Simple transplant by swapping positions in the list
        try:
            nodes = self.optimizer.tree_nodes
            source_idx = nodes.index(source)
            target_idx = nodes.index(target)
            nodes[source_idx], nodes[target_idx] = nodes[target_idx], nodes[source_idx]

            # Rebuild connections
            self.optimizer.rebuild_tree_connections()
        except (ValueError, IndexError):
            pass  # Ignore if transplant fails


class SimulatedAnnealingOptimizer:
    """Main class for simulated annealing optimization of block placement."""

    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.blocks = {}  # name -> OptimizationBlock
        self.tree_root = None
        self.tree_nodes = []
        self.symmetry_units = []
        self.nets = []
        self.placement_statistics = {}
        self.symmetry_groups = {}
        self.rng = random.Random()

        # Optimization parameters
        self.initial_temperature = 1000.0
        self.final_temperature = 0.1
        self.cooling_rate = 0.95
        self.max_iterations = 1000
        self.timeout_seconds = 30

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

    def extract_data_for_optimization(self):
        """Extract placement data and tree structure for optimization."""
        if not self.original_data or 'bstar_tree' not in self.original_data:
            print("No B* tree data found in JSON.")
            return False

        # Extract placement statistics
        if 'placement_statistics' in self.original_data['bstar_tree']:
            self.placement_statistics = self.original_data['bstar_tree']['placement_statistics']

        # Extract blocks and variants from original blocks section
        if 'blocks' in self.original_data:
            for block_data in self.original_data['blocks']:
                block = OptimizationBlock(block_data['name'])
                block.device_type = block_data.get('device_type', '')
                block.all_variants = block_data.get('variants', [])

                # Set current variant
                if block.all_variants:
                    default_variant = None
                    variant_index = 0
                    for i, variant in enumerate(block.all_variants):
                        if variant.get('is_default', False):
                            default_variant = variant
                            variant_index = i
                            break
                    if not default_variant:
                        default_variant = block.all_variants[0]
                        variant_index = 0

                    block.set_variant(default_variant, variant_index)

                self.blocks[block.name] = block

        # Extract nets for wirelength calculation
        if 'netlist' in self.original_data and 'nets' in self.original_data['netlist']:
            for net_data in self.original_data['netlist']['nets']:
                net_name = net_data['name']
                connected_instances = net_data.get('connected_instances', [])

                # Map instances to blocks
                net_blocks = []
                if 'netlist' in self.original_data and 'instances' in self.original_data['netlist']:
                    for instance_name in connected_instances:
                        for instance in self.original_data['netlist']['instances']:
                            if instance['name'] == instance_name:
                                block_name = instance.get('block', instance_name)
                                if block_name in self.blocks:
                                    net_blocks.append(self.blocks[block_name])
                                break

                if net_blocks:
                    self.nets.append((net_name, net_blocks))

        # Parse B* tree structure
        bstar_tree_data = self.original_data['bstar_tree']
        if 'root' in bstar_tree_data:
            self.tree_root = self.parse_tree_node(bstar_tree_data['root'])
            self.collect_tree_nodes(self.tree_root)

        # Extract symmetry information
        self.extract_symmetry_groups()

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

        # Parse units
        if 'units' in node_data:
            for unit_data in node_data['units']:
                l_half_data = unit_data.get('l_half', {})
                r_half_data = unit_data.get('r_half', {})

                l_half_name = l_half_data.get('name', '')
                r_half_name = r_half_data.get('name', '')

                l_half = self.blocks.get(l_half_name)
                r_half = self.blocks.get(r_half_name)

                if l_half and r_half:
                    # Update block positions from JSON
                    self.update_block_from_data(l_half, l_half_data)
                    self.update_block_from_data(r_half, r_half_data)

                    unit = OptimizationUnit(l_half, r_half)
                    node.units.append(unit)
                    self.symmetry_units.append(unit)

        # Parse child nodes
        node.x_child = self.parse_tree_node(node_data.get('x_child'))
        node.y_child = self.parse_tree_node(node_data.get('y_child'))

        return node

    def update_block_from_data(self, block, block_data):
        """Update block properties from JSON data."""
        block.width = block_data.get('width', block.width)
        block.height = block_data.get('height', block.height)
        block.x_min = block_data.get('x_min', 0.0)
        block.y_min = block_data.get('y_min', 0.0)
        block.x_max = block_data.get('x_max', block.x_min + block.width)
        block.y_max = block_data.get('y_max', block.y_min + block.height)
        block.color = block_data.get('color', block.color)

        if 'current_variant' in block_data:
            block.current_variant = block_data['current_variant']
            block.variant_index = block_data.get('variant_index', 0)

        block.is_placed = True

    def collect_tree_nodes(self, node):
        """Collect all tree nodes into a list for easy access."""
        if not node:
            return

        self.tree_nodes.append(node)
        self.collect_tree_nodes(node.x_child)
        self.collect_tree_nodes(node.y_child)

    def extract_symmetry_groups(self):
        """Extract symmetry group information."""
        if 'blocks' in self.original_data:
            for block_data in self.original_data['blocks']:
                if 'symmetry' in block_data:
                    symmetry = block_data['symmetry']
                    block_name = block_data['name']

                    if symmetry.get('type') == 'pair_symmetric':
                        pair_with = symmetry.get('pair_with')
                        if pair_with:
                            group_key = tuple(sorted([block_name, pair_with]))
                            self.symmetry_groups[group_key] = 'pair_symmetric'
                    elif symmetry.get('type') == 'self_symmetric':
                        group_key = (block_name,)
                        self.symmetry_groups[group_key] = 'self_symmetric'

    def get_symmetric_blocks(self, block):
        """Get all blocks that are symmetric to the given block."""
        symmetric_blocks = [block]

        for group_key, group_type in self.symmetry_groups.items():
            if block.name in group_key:
                for block_name in group_key:
                    if block_name != block.name and block_name in self.blocks:
                        symmetric_blocks.append(self.blocks[block_name])
                break

        return symmetric_blocks

    def rebuild_tree_connections(self):
        """Rebuild tree connections after structural changes."""
        if not self.tree_nodes:
            return

        # Rebuild main tree structure (x_child connections)
        for i in range(len(self.tree_nodes) - 1):
            self.tree_nodes[i].x_child = self.tree_nodes[i + 1]
            self.tree_nodes[i + 1].parent = self.tree_nodes[i]

        self.tree_nodes[-1].x_child = None
        if self.tree_nodes:
            self.tree_nodes[0].parent = None

    def perform_placement(self):
        """Perform placement based on current tree structure."""
        if not self.tree_root:
            return False

        # Reset all block positions
        for block in self.blocks.values():
            block.is_placed = False
            block.x_min = 0.0
            block.y_min = 0.0
            block.x_max = 0.0
            block.y_max = 0.0

        # Perform placement starting from root
        self.place_tree_node(self.tree_root, 0.0, 0.0)

        # Calculate placement statistics
        self.calculate_placement_statistics()

        return True

    def place_tree_node(self, node, x_offset=0.0, y_offset=0.0):
        """Places a tree node and its children recursively."""
        if not node:
            return {'x_min': x_offset, 'y_min': y_offset, 'x_max': x_offset, 'y_max': y_offset}

        placed_blocks = []

        # Place units within this node
        current_y = y_offset
        max_width = 0.0

        for unit in node.units:
            if unit.r_half == unit.l_half:
                # Single block (self-symmetric or individual)
                block = unit.r_half
                block.set_position(x_offset, current_y)
                placed_blocks.append(block)
                current_y += block.height
                max_width = max(max_width, block.width)
            else:
                # Symmetric pair
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
            node.x_max = x_offset
            node.y_max = y_offset

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

    def calculate_placement_statistics(self):
        """Calculate statistics about the current placement."""
        placed_blocks = [b for b in self.blocks.values() if b.is_placed]

        if not placed_blocks:
            self.placement_statistics = {
                "total_blocks": 0,
                "placed_blocks": 0,
                "total_block_area": 0.0,
                "bounding_box": {"width": 0.0, "height": 0.0, "area": 0.0},
                "utilization_percent": 0.0,
                "aspect_ratio": 1.0
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

    def save_state(self):
        """Save current state for rollback."""
        state = {
            'blocks_state': {},
            'tree_structure': copy.deepcopy(self.tree_nodes),
            'placement_statistics': copy.deepcopy(self.placement_statistics)
        }

        # Save block states
        for name, block in self.blocks.items():
            state['blocks_state'][name] = {
                'x_min': block.x_min,
                'y_min': block.y_min,
                'x_max': block.x_max,
                'y_max': block.y_max,
                'width': block.width,
                'height': block.height,
                'variant_index': block.variant_index,
                'current_variant': copy.deepcopy(block.current_variant),
                'is_placed': block.is_placed
            }

        return state

    def load_state(self, state):
        """Load saved state."""
        # Restore block states
        for name, block_state in state['blocks_state'].items():
            if name in self.blocks:
                block = self.blocks[name]
                block.x_min = block_state['x_min']
                block.y_min = block_state['y_min']
                block.x_max = block_state['x_max']
                block.y_max = block_state['y_max']
                block.width = block_state['width']
                block.height = block_state['height']
                block.variant_index = block_state['variant_index']
                block.current_variant = block_state['current_variant']
                block.is_placed = block_state['is_placed']

        # Restore tree structure
        self.tree_nodes = state['tree_structure']
        self.placement_statistics = state['placement_statistics']

    def optimize(self):
        """Perform simulated annealing optimization."""
        print("Starting simulated annealing optimization...")

        # Initialize cost function and perturbator
        cost_function = CostFunction()
        perturbator = Perturbator(self)
        timer = Timer(self.timeout_seconds)

        # Initial state
        current_cost = cost_function(self)
        best_cost = current_cost
        best_state = self.save_state()

        temperature = self.initial_temperature
        iteration = 0
        accepted_moves = 0
        rejected_moves = 0

        print(f"Initial cost: {current_cost:.6f}")

        while (temperature > self.final_temperature and
               iteration < self.max_iterations and
               not timer.is_timeout()):

            # Save current state
            current_state = self.save_state()

            # Apply perturbation
            perturbator()

            # Recalculate placement
            self.perform_placement()

            # Calculate new cost
            new_cost = cost_function(self)

            # Accept or reject move
            accept_move = False
            if new_cost < current_cost:
                # Always accept better solutions
                accept_move = True
            else:
                # Accept worse solutions with probability based on temperature
                delta_cost = new_cost - current_cost
                probability = math.exp(-delta_cost / temperature)
                if self.rng.random() < probability:
                    accept_move = True

            if accept_move:
                current_cost = new_cost
                accepted_moves += 1

                # Update best solution if needed
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = self.save_state()
                    print(f"Iteration {iteration}: New best cost: {best_cost:.6f}")
            else:
                # Reject move - restore previous state
                self.load_state(current_state)
                rejected_moves += 1

            # Cool down
            temperature *= self.cooling_rate
            iteration += 1

            # Print progress occasionally
            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: Current cost: {current_cost:.6f}, Best: {best_cost:.6f}, Temp: {temperature:.2f}")

        # Restore best solution
        self.load_state(best_state)
        self.perform_placement()

        print(f"\nOptimization completed:")
        print(f"  Total iterations: {iteration}")
        print(f"  Accepted moves: {accepted_moves}")
        print(f"  Rejected moves: {rejected_moves}")
        print(f"  Final best cost: {best_cost:.6f}")
        print(f"  Final temperature: {temperature:.6f}")

        return best_cost

    def update_tree_coordinates_in_json(self, node_data, placement_node):
        """Updates coordinates in the JSON tree structure."""
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
        """Updates the processed data with optimized placement information."""
        if not self.processed_data or 'bstar_tree' not in self.processed_data:
            return False

        # Update tree coordinates
        if 'root' in self.processed_data['bstar_tree'] and self.tree_root:
            self.update_tree_coordinates_in_json(
                self.processed_data['bstar_tree']['root'],
                self.tree_root
            )

        # Update placement statistics
        self.processed_data['bstar_tree']['placement_statistics'] = self.placement_statistics

        # Update description
        self.processed_data['bstar_tree']['description'] = "B* tree structure with optimized block placement"

        # Add optimization metadata
        self.processed_data['bstar_tree']['optimization_info'] = {
            "optimization_method": "simulated_annealing",
            "optimization_completed": True,
            "final_cost": float(CostFunction()(self)),
            "final_utilization": self.placement_statistics.get('utilization_percent', 0.0),
            "final_aspect_ratio": self.placement_statistics.get('aspect_ratio', 1.0)
        }

        return True

    def save_to_file(self, output_filename):
        """Saves processed data to JSON file."""
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
        """Outputs processed data to stdout (for n8n integration)."""
        if self.processed_data is not None:
            print(json.dumps(self.processed_data))
        else:
            print("No processed data available.", file=sys.stderr)

    def process_optimization(self):
        """Main processing method that performs optimization."""
        if not self.extract_data_for_optimization():
            return False

        if not self.optimize():
            return False

        if not self.update_processed_data():
            return False

        return True

    def main_n8n(self):
        """Main method for n8n integration."""
        if not self.load_from_n8n():
            print("Error: Could not load data from stdin", file=sys.stderr)
            return False

        if not self.process_optimization():
            print("Error: Could not perform optimization", file=sys.stderr)
            return False

        self.output_to_n8n()
        return True

    def main_local(self, input_filename, output_filename):
        """Main method for local file processing."""
        input_file = f"{input_filename}.json"
        output_file = f"{output_filename}.json"

        if not self.load_json_file(input_file):
            print(f"Error: Could not load input file {input_file}")
            return False

        if not self.process_optimization():
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

        return True

    def set_optimization_parameters(self, initial_temp=1000.0, final_temp=0.1,
                                    cooling_rate=0.95, max_iterations=1000,
                                    timeout_seconds=30):
        """Set optimization parameters."""
        self.initial_temperature = initial_temp
        self.final_temperature = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds


def optimize_placement_from_files(input_filename, output_filename):
    """
    Convenience function to optimize placement from input file and save to output file.

    Args:
        input_filename (str): Input JSON filename (without extension)
        output_filename (str): Output JSON filename (without extension)

    Returns:
        bool: True if successful, False otherwise
    """
    optimizer = SimulatedAnnealingOptimizer()

    # Set optimization parameters for quick testing
    optimizer.set_optimization_parameters(
        initial_temp=1000.0,
        final_temp=1.0,
        cooling_rate=0.95,
        max_iterations=500,
        timeout_seconds=30
    )

    success = optimizer.main_local(input_filename, output_filename)

    if success:
        print(f"Successfully optimized placement.")
        print(f"Input: {input_filename}.json")
        print(f"Output: {output_filename}.json")
    else:
        print(f"Failed to optimize placement.")

    return success


if __name__ == "__main__":
    # n8n integration mode only
    optimizer = SimulatedAnnealingOptimizer()
    optimizer.main_n8n()