import json
import sys
import random
import copy
import math


class PlacementBlock:
    """Represents a block for placement with position information."""

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
        self.device_type = ""
        self.is_placed = False
        self.color = "FFFFFF"  # Default white color

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


class SymmetryUnit:
    """Represents a symmetry unit for placement."""

    def __init__(self, r_half, l_half):
        self.r_half = r_half
        self.l_half = l_half
        self.x_child = None
        self.y_child = None
        self.parent = None


class PlacementTreeNode:
    """Represents a B* tree node for placement."""

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
        self.color = "FFFFFF"  # Default color for tree nodes


class InitialPlacer:
    """Main class for performing initial placement of blocks."""

    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.blocks = {}  # name -> PlacementBlock
        self.tree_root = None
        self.placement_statistics = {}
        self.rng = random.Random()
        self.symmetry_groups = {}  # Store symmetry group information
        self.used_colors = []  # Track used colors to prevent similar ones

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

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB hex string."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"

    def color_distance(self, color1, color2):
        """Calculate the perceptual distance between two colors in RGB space."""
        # Convert hex to RGB
        r1 = int(color1[0:2], 16)
        g1 = int(color1[2:4], 16)
        b1 = int(color1[4:6], 16)

        r2 = int(color2[0:2], 16)
        g2 = int(color2[2:4], 16)
        b2 = int(color2[4:6], 16)

        # Use weighted euclidean distance (accounts for human perception)
        dr = r1 - r2
        dg = g1 - g2
        db = b1 - b2

        # Weights based on human eye sensitivity
        return ((2 + (r1 + r2) / 256) * dr * dr +
                4 * dg * dg +
                (2 + (255 - r1 - r2) / 256) * db * db) ** 0.5

    def is_color_too_similar(self, new_color, min_distance=80):
        """Check if a color is too similar to existing colors."""
        for existing_color in self.used_colors:
            if self.color_distance(new_color, existing_color) < min_distance:
                return True
        return False

    def generate_random_color(self):
        """Generate a random hex color from the full spectrum with good separation."""
        max_attempts = 100

        for attempt in range(max_attempts):
            # Use HSV color space for better distribution across the spectrum
            # Hue: 0-360 degrees (full spectrum)
            hue = self.rng.uniform(0, 1)

            # Saturation: 0.6-1.0 (avoid washed out colors)
            saturation = self.rng.uniform(0.6, 1.0)

            # Value/Brightness: 0.5-0.9 (avoid too dark or too bright)
            value = self.rng.uniform(0.5, 0.9)

            # Convert to hex
            color = self.hsv_to_rgb(hue, saturation, value)

            # Check if this color is sufficiently different from existing ones
            if not self.is_color_too_similar(color):
                self.used_colors.append(color)
                return color

        # Fallback: if we can't find a sufficiently different color after max_attempts,
        # generate a color anyway (better than infinite loop)
        fallback_color = self.hsv_to_rgb(
            self.rng.uniform(0, 1),
            self.rng.uniform(0.6, 1.0),
            self.rng.uniform(0.5, 0.9)
        )
        self.used_colors.append(fallback_color)
        return fallback_color

    def generate_distributed_colors(self, count):
        """Generate a set of well-distributed colors across the spectrum."""
        colors = []

        if count <= 0:
            return colors

        # Use golden ratio for better distribution
        golden_ratio = 0.618033988749895

        for i in range(count):
            # Distribute hues evenly with golden ratio offset
            hue = (i * golden_ratio) % 1.0

            # Vary saturation and value slightly for each color
            saturation = 0.7 + 0.3 * self.rng.uniform(0, 1)  # 0.7-1.0
            value = 0.6 + 0.3 * self.rng.uniform(0, 1)  # 0.6-0.9

            color = self.hsv_to_rgb(hue, saturation, value)
            colors.append(color)

        return colors

    def assign_colors_to_blocks(self):
        """Assign random colors to blocks while ensuring symmetric blocks have the same color."""
        # First, count how many unique colors we need
        color_groups = []

        # Identify symmetry groups from original block data
        if 'blocks' in self.original_data:
            for block_data in self.original_data['blocks']:
                if 'symmetry' in block_data:
                    symmetry = block_data['symmetry']
                    block_name = block_data['name']

                    if symmetry.get('type') == 'pair_symmetric':
                        pair_with = symmetry.get('pair_with')
                        if pair_with:
                            # Create symmetry group
                            group_key = tuple(sorted([block_name, pair_with]))
                            if group_key not in self.symmetry_groups:
                                self.symmetry_groups[group_key] = None  # Will assign color later
                                color_groups.append(group_key)

                    elif symmetry.get('type') == 'self_symmetric':
                        # Self-symmetric blocks get their own color
                        group_key = (block_name,)
                        if group_key not in self.symmetry_groups:
                            self.symmetry_groups[group_key] = None  # Will assign color later
                            color_groups.append(group_key)

        # Count individual blocks (not in symmetry groups)
        individual_blocks = []
        for block_name in self.blocks.keys():
            is_in_group = False
            for group_key in self.symmetry_groups.keys():
                if block_name in group_key:
                    is_in_group = True
                    break
            if not is_in_group:
                individual_blocks.append(block_name)

        # Generate well-distributed colors
        total_colors_needed = len(color_groups) + len(individual_blocks)

        if total_colors_needed > 0:
            # Use distributed color generation for better results
            distributed_colors = self.generate_distributed_colors(total_colors_needed)
            self.rng.shuffle(distributed_colors)  # Shuffle to randomize assignment

            color_index = 0

            # Assign colors to symmetry groups
            for group_key in color_groups:
                if color_index < len(distributed_colors):
                    self.symmetry_groups[group_key] = distributed_colors[color_index]
                    color_index += 1
                else:
                    # Fallback to random generation if we run out
                    self.symmetry_groups[group_key] = self.generate_random_color()

            # Assign colors to blocks based on symmetry groups
            for block_name, block in self.blocks.items():
                color_assigned = False

                # Check if this block is part of any symmetry group
                for group_key, color in self.symmetry_groups.items():
                    if block_name in group_key:
                        block.color = color
                        color_assigned = True
                        break

                # If not part of any symmetry group, assign individual color
                if not color_assigned:
                    if color_index < len(distributed_colors):
                        block.color = distributed_colors[color_index]
                        color_index += 1
                    else:
                        # Fallback to random generation
                        block.color = self.generate_random_color()

    def extract_blocks_and_tree(self):
        """Extracts blocks and B* tree structure from JSON data."""
        if not self.original_data or 'bstar_tree' not in self.original_data:
            print("No B* tree data found in JSON.")
            return False

        # Extract blocks information from original blocks section
        if 'blocks' in self.original_data:
            for block_data in self.original_data['blocks']:
                block = PlacementBlock(block_data['name'])
                block.device_type = block_data.get('device_type', '')

                # Set default variant
                if 'variants' in block_data and block_data['variants']:
                    default_variant = None
                    variant_index = 0
                    for i, variant in enumerate(block_data['variants']):
                        if variant.get('is_default', False):
                            default_variant = variant
                            variant_index = i
                            break
                    if not default_variant:
                        default_variant = block_data['variants'][0]
                        variant_index = 0

                    block.set_variant(default_variant, variant_index)

                self.blocks[block.name] = block

        # Assign colors to blocks based on symmetry
        self.assign_colors_to_blocks()

        # Parse B* tree structure
        bstar_tree_data = self.original_data['bstar_tree']
        if 'root' in bstar_tree_data:
            self.tree_root = self.parse_tree_node(bstar_tree_data['root'])

        return True

    def parse_tree_node(self, node_data):
        """Recursively parse B* tree node from JSON data."""
        if not node_data:
            return None

        node = PlacementTreeNode(node_data['name'])

        # Assign color to tree node based on its units
        if 'units' in node_data and node_data['units']:
            # Use the color of the first unit's block
            first_unit = node_data['units'][0]
            if 'l_half' in first_unit:
                block_name = first_unit['l_half']['name']
                if block_name in self.blocks:
                    node.color = self.blocks[block_name].color
        else:
            # If no units, assign random color
            node.color = self.generate_random_color()

        # Parse units
        if 'units' in node_data:
            for unit_data in node_data['units']:
                l_half_name = unit_data['l_half']['name']
                r_half_name = unit_data['r_half']['name']

                l_half = self.blocks.get(l_half_name)
                r_half = self.blocks.get(r_half_name)

                if l_half and r_half:
                    unit = SymmetryUnit(r_half, l_half)

                    # Parse child nodes for units
                    unit.x_child = self.parse_tree_node(unit_data.get('x_child'))
                    unit.y_child = self.parse_tree_node(unit_data.get('y_child'))

                    node.units.append(unit)

        # Parse child nodes for tree node
        node.x_child = self.parse_tree_node(node_data.get('x_child'))
        node.y_child = self.parse_tree_node(node_data.get('y_child'))

        return node

    def perform_initial_placement(self):
        """Performs initial placement of blocks based on B* tree structure."""
        if not self.tree_root:
            print("No tree root available for placement.")
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
        """Calculates statistics about the placement."""
        placed_blocks = [b for b in self.blocks.values() if b.is_placed]

        if not placed_blocks:
            self.placement_statistics = {
                "total_blocks": 0,
                "placed_blocks": 0,
                "total_area": 0.0,
                "bounding_box": {"width": 0.0, "height": 0.0, "area": 0.0},
                "utilization": 0.0
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
        """Updates the processed data with placement information."""
        if not self.processed_data or 'bstar_tree' not in self.processed_data:
            return False

        # Update tree coordinates
        if 'root' in self.processed_data['bstar_tree'] and self.tree_root:
            self.update_tree_coordinates_in_json(
                self.processed_data['bstar_tree']['root'],
                self.tree_root
            )

        # Add placement statistics to bstar_tree
        self.processed_data['bstar_tree']['placement_statistics'] = self.placement_statistics

        # Update description
        self.processed_data['bstar_tree']['description'] = "B* tree structure with initial block placement"

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

    def process_placement(self):
        """Main processing method that performs initial placement."""
        if not self.extract_blocks_and_tree():
            return False

        if not self.perform_initial_placement():
            return False

        if not self.update_processed_data():
            return False

        return True

    def main_n8n(self):
        """Main method for n8n integration."""
        self.load_from_n8n()
        self.process_placement()
        self.output_to_n8n()

    def main_local(self, input_filename, output_filename):
        """Main method for local file processing."""
        input_file = f"{input_filename}.json"
        output_file = f"{output_filename}.json"

        if not self.load_json_file(input_file):
            print(f"Error: Could not load input file {input_file}")
            return False

        if not self.process_placement():
            print(f"Error: Could not perform initial placement")
            return False

        if not self.save_to_file(output_file):
            print(f"Error: Could not save output file {output_file}")
            return False

        print(f"Successfully performed initial placement:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Placed {self.placement_statistics['placed_blocks']} blocks")
        print(
            f"  Bounding box: {self.placement_statistics['bounding_box']['width']:.2f} x {self.placement_statistics['bounding_box']['height']:.2f}")
        print(f"  Utilization: {self.placement_statistics['utilization_percent']:.2f}%")
        print(f"  Aspect ratio: {self.placement_statistics['aspect_ratio']:.3f}")

        return True


def perform_initial_placement_from_files(input_filename, output_filename):
    """
    Convenience function to perform initial placement from input file and save to output file.

    Args:
        input_filename (str): Input JSON filename (without extension)
        output_filename (str): Output JSON filename (without extension)

    Returns:
        bool: True if successful, False otherwise
    """
    placer = InitialPlacer()
    input_file = f"{input_filename}.json"
    output_file = f"{output_filename}.json"

    success = placer.main_local(input_filename, output_filename)

    if success:
        print(f"Successfully completed initial placement.")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
    else:
        print(f"Failed to perform initial placement.")

    return success


if __name__ == "__main__":
    # n8n integration mode only
    placer = InitialPlacer()
    placer.main_n8n()