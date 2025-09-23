import json
import sys
import copy
import random
import math
import time


class PlacementBlock:
    """Block representation compatible with tree2.py structure."""

    def __init__(self, name):
        self.name = name
        self.width = 0.0
        self.height = 0.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.variants = []
        self.current_variant = None
        self.variant_index = 0
        self.merge_device_names = []
        self.device_type = ""

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_center_x(self):
        return (self.x_min + self.x_max) * 0.5

    def get_center_y(self):
        return (self.y_min + self.y_max) * 0.5

    def set_position(self, x_min, y_min):
        """Set block position maintaining width/height."""
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_min + self.width
        self.y_max = y_min + self.height

    def set_variant(self, variant, variant_index=0):
        """Set block variant and update dimensions."""
        if variant:
            self.width = variant['width']
            self.height = variant['height']
            self.current_variant = variant
            self.variant_index = variant_index
            # Update x_max and y_max if already positioned
            if hasattr(self, 'x_min'):
                self.x_max = self.x_min + self.width
                self.y_max = self.y_min + self.height


class OptimizationNet:
    """Network representation for HPWL calculation."""

    def __init__(self, name, blocks):
        self.name = name
        self.blocks = blocks


class SimulatedAnnealingOptimizer:
    """
    Enhanced Simulated Annealing Optimizer that uses tree2.py optimization principles
    with proper block placement and cost calculation.
    """

    def __init__(self):
        self.placement_data = None
        self.original_data = None

        # Optimization parameters
        self.initial_temp = 2000.0
        self.final_temp = 1.0
        self.cooling_rate = 0.98
        self.max_iterations = 2000
        self.timeout_seconds = 120
        self.patience = 500
        self.aspect_ratio_target = 2.0

        # Internal state
        self.current_temp = self.initial_temp
        self.iteration = 0
        self.start_time = None
        self.rng = random.Random()

        # Placement data
        self.all_nodes = []
        self.blocks = {}  # name -> PlacementBlock
        self.nets = []  # List of OptimizationNet

        # Bounding box
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0

    def set_optimization_parameters(self, initial_temp=2000.0, final_temp=1.0,
                                    cooling_rate=0.98, max_iterations=2000,
                                    timeout_seconds=120, patience=500,
                                    aspect_ratio_target=2.0):
        """Set optimization parameters for simulated annealing."""
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.patience = patience
        self.aspect_ratio_target = aspect_ratio_target
        self.current_temp = self.initial_temp

    def load_json_file(self, input_filename):
        """Load JSON file and extract placement data."""
        filename = input_filename if input_filename.endswith('.json') else input_filename + '.json'
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.original_data = copy.deepcopy(data)
        self.placement_data = copy.deepcopy(data)

        self._extract_placement_data()
        return self.placement_data

    def load_from_n8n(self):
        """Load JSON data from stdin."""
        data = json.load(sys.stdin)
        self.original_data = copy.deepcopy(data)
        self.placement_data = copy.deepcopy(data)

        self._extract_placement_data()
        return self.placement_data

    def _extract_placement_data(self):
        """Extract blocks, nets, and tree structure from JSON."""
        # Extract blocks
        self.blocks = {}
        if 'blocks' in self.placement_data:
            for block_data in self.placement_data['blocks']:
                block = PlacementBlock(block_data['name'])
                block.device_type = block_data.get('device_type', '')
                block.variants = block_data.get('variants', [])

                # Set default variant
                if block.variants:
                    default_variant = None
                    for i, variant in enumerate(block.variants):
                        if variant.get('is_default', False):
                            default_variant = variant
                            block.variant_index = i
                            break
                    if not default_variant:
                        default_variant = block.variants[0]
                        block.variant_index = 0

                    block.set_variant(default_variant, block.variant_index)

                self.blocks[block.name] = block

        # Extract nets
        self._extract_nets()

        # Extract tree structure
        self.all_nodes = []
        if 'bstar_tree' in self.placement_data and 'root' in self.placement_data['bstar_tree']:
            self._collect_tree_nodes(self.placement_data['bstar_tree']['root'])

    def _extract_nets(self):
        """Extract network connections for HPWL calculation."""
        self.nets = []
        if 'netlist' not in self.placement_data or 'nets' not in self.placement_data['netlist']:
            return

        # Create mapping from instance to block
        instance_to_block = {}
        if 'instances' in self.placement_data['netlist']:
            for instance in self.placement_data['netlist']['instances']:
                instance_name = instance['name']
                block_name = instance.get('block', instance_name)
                if block_name in self.blocks:
                    instance_to_block[instance_name] = self.blocks[block_name]

        # Create nets
        for net_data in self.placement_data['netlist']['nets']:
            net_name = net_data['name']
            connected_instances = net_data.get('connected_instances', [])

            # Map instances to blocks
            net_blocks = []
            for instance_name in connected_instances:
                if instance_name in instance_to_block:
                    block = instance_to_block[instance_name]
                    if block not in net_blocks:  # Avoid duplicates
                        net_blocks.append(block)

            if len(net_blocks) > 1:  # Only nets with multiple blocks
                self.nets.append(OptimizationNet(net_name, net_blocks))

    def _collect_tree_nodes(self, node):
        """Recursively collect all nodes in the tree."""
        if not node:
            return

        self.all_nodes.append(node)

        if node.get('x_child'):
            self._collect_tree_nodes(node['x_child'])
        if node.get('y_child'):
            self._collect_tree_nodes(node['y_child'])

    def _place_blocks_from_tree(self):
        """Place all blocks according to B* tree structure."""
        # Reset all block positions
        for block in self.blocks.values():
            block.x_min = 0.0
            block.y_min = 0.0
            block.x_max = 0.0
            block.y_max = 0.0

        # Place blocks starting from root
        if 'bstar_tree' in self.placement_data and 'root' in self.placement_data['bstar_tree']:
            root = self.placement_data['bstar_tree']['root']
            placed_blocks = self._place_node(root, 0.0, 0.0)
            self._update_bounding_box(placed_blocks)

        return list(self.blocks.values())

    def _place_node(self, node, x_offset, y_offset):
        """Place a node and its children recursively."""
        if not node:
            return []

        placed_blocks = []
        current_y = y_offset
        max_width = 0.0

        # Place units within this node
        if 'units' in node:
            for unit in node['units']:
                unit_blocks = []

                # Extract blocks from l_half and r_half
                if 'l_half' in unit:
                    l_half_name = unit['l_half']['name']
                    if l_half_name in self.blocks:
                        l_block = self.blocks[l_half_name]
                        # Update block variant if different from JSON
                        if 'current_variant' in unit['l_half']:
                            variant = unit['l_half']['current_variant']
                            variant_idx = unit['l_half'].get('variant_index', 0)
                            l_block.set_variant(variant, variant_idx)
                        unit_blocks.append(l_block)

                if 'r_half' in unit:
                    r_half_name = unit['r_half']['name']
                    if r_half_name in self.blocks and r_half_name != unit.get('l_half', {}).get('name'):
                        r_block = self.blocks[r_half_name]
                        # Update block variant if different from JSON
                        if 'current_variant' in unit['r_half']:
                            variant = unit['r_half']['current_variant']
                            variant_idx = unit['r_half'].get('variant_index', 0)
                            r_block.set_variant(variant, variant_idx)
                        unit_blocks.append(r_block)

                # Place blocks in the unit
                if len(unit_blocks) == 1:
                    # Single block (self-symmetric or individual)
                    block = unit_blocks[0]
                    block.set_position(x_offset, current_y)
                    placed_blocks.append(block)
                    current_y += block.height
                    max_width = max(max_width, block.width)

                elif len(unit_blocks) == 2:
                    # Symmetric pair
                    l_block, r_block = unit_blocks[0], unit_blocks[1]

                    # Place left block
                    l_block.set_position(x_offset, current_y)
                    # Place right block next to left block
                    r_block.set_position(x_offset + l_block.width, current_y)

                    placed_blocks.extend([l_block, r_block])
                    current_y += max(l_block.height, r_block.height)
                    max_width = max(max_width, l_block.width + r_block.width)

        # Calculate this node's bounding box
        if placed_blocks:
            node_x_min = min(block.x_min for block in placed_blocks)
            node_y_min = min(block.y_min for block in placed_blocks)
            node_x_max = max(block.x_max for block in placed_blocks)
            node_y_max = max(block.y_max for block in placed_blocks)
        else:
            node_x_min = x_offset
            node_y_min = y_offset
            node_x_max = x_offset
            node_y_max = y_offset

        # Place x_child (horizontally adjacent)
        if node.get('x_child'):
            x_child_blocks = self._place_node(node['x_child'], node_x_max, node_y_min)
            placed_blocks.extend(x_child_blocks)
            if x_child_blocks:
                node_x_max = max(node_x_max, max(block.x_max for block in x_child_blocks))
                node_y_max = max(node_y_max, max(block.y_max for block in x_child_blocks))

        # Place y_child (vertically adjacent)
        if node.get('y_child'):
            y_child_blocks = self._place_node(node['y_child'], node_x_min, node_y_max)
            placed_blocks.extend(y_child_blocks)
            if y_child_blocks:
                node_x_max = max(node_x_max, max(block.x_max for block in y_child_blocks))
                node_y_max = max(node_y_max, max(block.y_max for block in y_child_blocks))

        return placed_blocks

    def _update_bounding_box(self, blocks):
        """Update global bounding box."""
        if blocks:
            self.x_min = min(block.x_min for block in blocks)
            self.y_min = min(block.y_min for block in blocks)
            self.x_max = max(block.x_max for block in blocks)
            self.y_max = max(block.y_max for block in blocks)
        else:
            self.x_min = self.y_min = self.x_max = self.y_max = 0.0

    def _calculate_cost(self):
        """Calculate optimization cost similar to tree2.py."""
        # Place blocks to get current positions
        blocks = self._place_blocks_from_tree()

        # Calculate HPWL (Half Perimeter Wire Length)
        total_hpwl = 0.0
        for net in self.nets:
            if len(net.blocks) < 2:
                continue

            x_coords = [block.get_center_x() for block in net.blocks]
            y_coords = [block.get_center_y() for block in net.blocks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            hpwl = (x_max - x_min) + (y_max - y_min)
            total_hpwl += hpwl

        # Calculate area and aspect ratio
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        area = width * height

        # Aspect ratio penalty (target is self.aspect_ratio_target)
        aspect_ratio = height / width if width > 0 else float('inf')
        aspect_penalty = abs(aspect_ratio - self.aspect_ratio_target)

        # Combined cost (weights similar to tree2.py)
        cost = total_hpwl + 0.1 * area + 100.0 * aspect_penalty

        return cost

    def _apply_random_perturbation(self):
        """Apply random perturbation to the tree structure."""
        perturbation_types = [
            self._swap_children,
            self._change_block_variants,
            self._restructure_tree,
            self._shuffle_units
        ]

        # Apply 1-2 perturbations
        num_perturbations = self.rng.randint(1, 2)
        for _ in range(num_perturbations):
            if self.all_nodes:
                perturbation = self.rng.choice(perturbation_types)
                perturbation()

    def _swap_children(self):
        """Randomly swap x_child and y_child of nodes."""
        if not self.all_nodes:
            return

        num_swaps = self.rng.randint(1, min(3, len(self.all_nodes)))
        selected_nodes = self.rng.sample(self.all_nodes, num_swaps)

        for node in selected_nodes:
            if node.get('x_child') or node.get('y_child'):
                x_child = node.get('x_child')
                y_child = node.get('y_child')
                node['x_child'] = y_child
                node['y_child'] = x_child

    def _change_block_variants(self):
        """Randomly change block variants."""
        changeable_blocks = [block for block in self.blocks.values()
                             if len(block.variants) > 1]

        if not changeable_blocks:
            return

        num_changes = self.rng.randint(1, min(2, len(changeable_blocks)))
        selected_blocks = self.rng.sample(changeable_blocks, num_changes)

        for block in selected_blocks:
            new_variant_idx = self.rng.randint(0, len(block.variants) - 1)
            if new_variant_idx != block.variant_index:
                new_variant = block.variants[new_variant_idx]
                block.set_variant(new_variant, new_variant_idx)

                # Update corresponding JSON nodes
                self._update_json_variants(block.name, new_variant, new_variant_idx)

    def _update_json_variants(self, block_name, variant, variant_idx):
        """Update JSON tree with new variant information."""
        for node in self.all_nodes:
            if 'units' not in node:
                continue
            for unit in node['units']:
                for half_name in ['l_half', 'r_half']:
                    if half_name in unit and unit[half_name]['name'] == block_name:
                        unit[half_name]['width'] = variant['width']
                        unit[half_name]['height'] = variant['height']
                        unit[half_name]['current_variant'] = variant
                        unit[half_name]['variant_index'] = variant_idx

    def _restructure_tree(self):
        """Randomly restructure tree topology."""
        if len(self.all_nodes) < 3:
            return

        # Find nodes with children to restructure
        internal_nodes = [node for node in self.all_nodes
                          if node.get('x_child') or node.get('y_child')]

        if not internal_nodes:
            return

        node = self.rng.choice(internal_nodes)

        # Simple restructuring: move child between x and y positions
        if node.get('x_child') and not node.get('y_child'):
            if self.rng.random() < 0.3:
                node['y_child'] = node['x_child']
                node['x_child'] = None
        elif node.get('y_child') and not node.get('x_child'):
            if self.rng.random() < 0.3:
                node['x_child'] = node['y_child']
                node['y_child'] = None

    def _shuffle_units(self):
        """Randomly shuffle units within nodes."""
        nodes_with_units = [node for node in self.all_nodes
                            if 'units' in node and len(node['units']) > 1]

        if not nodes_with_units:
            return

        node = self.rng.choice(nodes_with_units)
        self.rng.shuffle(node['units'])

    def _accept_move(self, delta_cost):
        """Determine whether to accept a move based on simulated annealing."""
        if delta_cost <= 0:
            return True

        if self.current_temp <= 0:
            return False

        probability = math.exp(-delta_cost / self.current_temp)
        return self.rng.random() < probability

    def _update_temperature(self):
        """Update temperature using cooling schedule."""
        self.current_temp = max(self.final_temp,
                                self.current_temp * self.cooling_rate)

    def _is_timeout(self):
        """Check if optimization should stop due to timeout."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout_seconds

    def _save_state(self):
        """Save current state."""
        return {
            'placement_data': copy.deepcopy(self.placement_data),
            'blocks': {name: copy.deepcopy(block.__dict__) for name, block in self.blocks.items()}
        }

    def _restore_state(self, state):
        """Restore previous state."""
        self.placement_data = copy.deepcopy(state['placement_data'])

        # Restore block states
        for name, block_dict in state['blocks'].items():
            if name in self.blocks:
                for attr, value in block_dict.items():
                    setattr(self.blocks[name], attr, value)

        # Re-extract tree nodes
        self.all_nodes = []
        if 'bstar_tree' in self.placement_data and 'root' in self.placement_data['bstar_tree']:
            self._collect_tree_nodes(self.placement_data['bstar_tree']['root'])

    def optimize_tree_structure(self):
        """Main optimization loop using simulated annealing."""
        print(f"Starting simulated annealing optimization...")
        print(f"Parameters: T0={self.initial_temp}, Tf={self.final_temp}, "
              f"cooling={self.cooling_rate}, max_iter={self.max_iterations}")
        print(f"Aspect ratio target: {self.aspect_ratio_target}")

        self.start_time = time.time()
        self.current_temp = self.initial_temp

        # Calculate initial cost
        current_cost = self._calculate_cost()
        best_cost = current_cost

        # Save initial state
        best_state = self._save_state()

        no_improvement_count = 0
        accepted_moves = 0

        print(f"Initial cost: {current_cost:.6f}")

        for iteration in range(self.max_iterations):
            self.iteration = iteration

            if self._is_timeout():
                print(f"Optimization stopped due to timeout at iteration {iteration}")
                break

            if no_improvement_count >= self.patience:
                print(f"Optimization stopped due to patience limit at iteration {iteration}")
                break

            # Save current state before perturbation
            prev_state = self._save_state()

            # Apply random perturbation
            self._apply_random_perturbation()

            # Calculate new cost
            new_cost = self._calculate_cost()
            delta_cost = new_cost - current_cost

            if self._accept_move(delta_cost):
                # Accept the move
                current_cost = new_cost
                accepted_moves += 1

                if new_cost < best_cost:
                    # New best solution
                    best_cost = new_cost
                    best_state = self._save_state()
                    no_improvement_count = 0
                    print(f"Iteration {iteration}: New best cost {best_cost:.6f} "
                          f"(T={self.current_temp:.2f})")
                else:
                    no_improvement_count += 1
            else:
                # Reject the move - restore previous state
                self._restore_state(prev_state)
                no_improvement_count += 1

            # Update temperature
            self._update_temperature()

            # Progress report
            if iteration % 200 == 0 and iteration > 0:
                acceptance_rate = accepted_moves / (iteration + 1) * 100
                print(f"Iteration {iteration}/{self.max_iterations}, "
                      f"Cost: {current_cost:.6f}, Best: {best_cost:.6f}, "
                      f"T={self.current_temp:.3f}, "
                      f"Accept: {acceptance_rate:.1f}%, "
                      f"No improvement: {no_improvement_count}")

        # Restore best state
        self._restore_state(best_state)

        # Final placement to update JSON coordinates
        self._place_blocks_from_tree()
        self._update_json_coordinates()

        print(f"Optimization completed after {iteration + 1} iterations")
        print(f"Final cost: {best_cost:.6f}")
        print(f"Bounding box: {self.x_max - self.x_min:.2f} x {self.y_max - self.y_min:.2f}")

        return self.placement_data

    def _update_json_coordinates(self):
        """Update JSON with final block coordinates."""
        for node in self.all_nodes:
            if 'units' not in node:
                continue

            for unit in node['units']:
                for half_name in ['l_half', 'r_half']:
                    if half_name in unit:
                        block_name = unit[half_name]['name']
                        if block_name in self.blocks:
                            block = self.blocks[block_name]
                            unit[half_name]['x_min'] = round(block.x_min, 6)
                            unit[half_name]['y_min'] = round(block.y_min, 6)
                            unit[half_name]['x_max'] = round(block.x_max, 6)
                            unit[half_name]['y_max'] = round(block.y_max, 6)
                            unit[half_name]['width'] = round(block.width, 6)
                            unit[half_name]['height'] = round(block.height, 6)

        # Update placement statistics
        placed_blocks = list(self.blocks.values())
        total_block_area = sum(b.width * b.height for b in placed_blocks)
        bbox_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        utilization = (total_block_area / bbox_area * 100) if bbox_area > 0 else 0

        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        aspect_ratio = height / width if width > 0 else 0

        if 'bstar_tree' in self.placement_data:
            self.placement_data['bstar_tree']['placement_statistics'] = {
                "total_blocks": len(self.blocks),
                "placed_blocks": len(placed_blocks),
                "total_block_area": round(total_block_area, 6),
                "bounding_box": {
                    "x_min": round(self.x_min, 6),
                    "y_min": round(self.y_min, 6),
                    "x_max": round(self.x_max, 6),
                    "y_max": round(self.y_max, 6),
                    "width": round(width, 6),
                    "height": round(height, 6),
                    "area": round(bbox_area, 6)
                },
                "utilization_percent": round(utilization, 2),
                "aspect_ratio": round(aspect_ratio, 3)
            }

    def main_local(self, input_file, output_file, *args, **kwargs):
        """Main method for local file processing."""
        # Load JSON data
        self.load_json_file(input_file)

        # Run optimization
        optimized_data = self.optimize_tree_structure()

        # Save to output file if specified
        if output_file:
            outname = output_file if output_file.endswith('.json') else output_file + '.json'
            with open(outname, 'w', encoding='utf-8') as f:
                json.dump(optimized_data, f, indent=2, ensure_ascii=False)
            print(f"Optimized tree saved to: {outname}")

        return optimized_data

    def main_n8n(self):
        """Main method for n8n integration."""
        # Load from stdin
        self.load_from_n8n()

        # Run optimization
        optimized_data = self.optimize_tree_structure()

        # Output to stdout
        print(json.dumps(optimized_data, indent=2, ensure_ascii=False))

        return optimized_data


if __name__ == "__main__":
    # Test with local file
    optimizer = SimulatedAnnealingOptimizer()

    # Set optimization parameters (matching mainControlFile.py)
    optimizer.set_optimization_parameters(
        initial_temp=2000.0,
        final_temp=1.0,
        cooling_rate=0.98,
        max_iterations=2000,
        timeout_seconds=120,
        patience=500,
        aspect_ratio_target=2.0
    )

    # Run optimization
    optimizer.main_local("initPlacement_out01", "optimizedPlacement_out01")