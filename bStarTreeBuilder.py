import json
import sys
import random
import copy


class HardBlock:
    """Represents a hard block for B* tree construction."""

    def __init__(self, name):
        self.name = name
        self.merge_device_names = []
        self.variants = []
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.width = 0.0
        self.height = 0.0
        self.column_multiple = 1
        self.row_multiple = 1
        self.device_type = ""
        self.pins = {}
        self.params = {}
        self.symmetry = {}

    def get_width(self):
        return self.x_max - self.x_min if self.x_max > self.x_min else self.width

    def get_height(self):
        return self.y_max - self.y_min if self.y_max > self.y_min else self.height

    def set_width(self, w):
        self.width = w
        self.x_max = self.x_min + w

    def set_height(self, h):
        self.height = h
        self.y_max = self.y_min + h

    def get_center_x(self):
        return (self.x_min + self.x_max) * 0.5

    def get_center_y(self):
        return (self.y_min + self.y_max) * 0.5


class SymmetryUnit:
    """Represents a symmetry unit in the B* tree."""

    def __init__(self, r_half, l_half):
        self.r_half = r_half
        self.l_half = l_half
        self.x_child = None
        self.y_child = None
        self.parent = None

        if r_half != l_half:
            if hasattr(l_half, 'variants') and l_half.variants:
                l_half.set_width(l_half.variants[0]['width'] if l_half.variants else l_half.width)
                l_half.set_height(l_half.variants[0]['height'] if l_half.variants else l_half.height)
            if hasattr(r_half, 'variants') and r_half.variants:
                r_half.set_width(r_half.variants[0]['width'] if r_half.variants else r_half.width)
                r_half.set_height(r_half.variants[0]['height'] if r_half.variants else r_half.height)


class TopologyBStarTree:
    """Represents a B* tree topology node."""

    def __init__(self, block_or_group, group=None):
        if group is not None:
            self.name = str(block_or_group)
            self.units = [SymmetryUnit(b1, b2) for b1, b2 in group]
        elif isinstance(block_or_group, HardBlock):
            self.name = block_or_group.name
            self.units = [SymmetryUnit(block_or_group, block_or_group)]
        else:
            self.name = str(block_or_group)
            self.units = [SymmetryUnit(b1, b2) for b1, b2 in block_or_group]

        self.root = self.units[0] if self.units else None
        self.x_child = None
        self.y_child = None
        self.parent = None
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0


class BStarTreeBuilder:
    """Main class for building B* trees from JSON input."""

    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.blocks = []
        self.symmetry_groups = {}
        self.bstar_tree_nodes = []
        self.root = None
        self.rng = random.Random()

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

    def extract_blocks_from_json(self):
        """Extracts blocks from JSON data and converts them to HardBlock objects."""
        if not self.original_data or 'blocks' not in self.original_data:
            print("No blocks data found in JSON.")
            return []

        self.blocks = []
        for block_data in self.original_data['blocks']:
            block = HardBlock(block_data['name'])

            # Extract basic properties
            if 'merge_device_names' in block_data:
                block.merge_device_names = block_data['merge_device_names']

            if 'variants' in block_data:
                block.variants = block_data['variants']
                # Use default variant (first one or the one marked as default)
                default_variant = None
                for variant in block.variants:
                    if variant.get('is_default', False):
                        default_variant = variant
                        break
                if not default_variant and block.variants:
                    default_variant = block.variants[0]

                if default_variant:
                    block.width = default_variant['width']
                    block.height = default_variant['height']
                    block.column_multiple = default_variant.get('column_multiple', 1)
                    block.row_multiple = default_variant.get('row_multiple', 1)

            if 'device_type' in block_data:
                block.device_type = block_data['device_type']

            if 'pins' in block_data:
                block.pins = block_data['pins']

            if 'params' in block_data:
                block.params = block_data['params']

            if 'symmetry' in block_data:
                block.symmetry = block_data['symmetry']
                # Group blocks by symmetry constraints
                sym_type = block_data['symmetry'].get('type')
                if sym_type == 'pair_symmetric':
                    pair_with = block_data['symmetry'].get('pair_with')
                    if pair_with:
                        group_name = f"sym_{block.name}_{pair_with}"
                        if group_name not in self.symmetry_groups:
                            self.symmetry_groups[group_name] = []
                        self.symmetry_groups[group_name].append(block)

            self.blocks.append(block)

        return self.blocks

    def build_bstar_tree(self):
        """Builds B* tree from extracted blocks."""
        if not self.blocks:
            print("No blocks available for tree construction.")
            return None

        # Create block name to block mapping
        block_name_to_block = {block.name: block for block in self.blocks}

        # Handle symmetry constraints
        constrained_blocks = set()
        tree_nodes = []

        # Process symmetry groups
        processed_pairs = set()
        for block in self.blocks:
            if hasattr(block, 'symmetry') and block.symmetry:
                sym_type = block.symmetry.get('type')

                if sym_type == 'pair_symmetric':
                    pair_with = block.symmetry.get('pair_with')
                    if pair_with and pair_with in block_name_to_block:
                        pair_key = tuple(sorted([block.name, pair_with]))
                        if pair_key not in processed_pairs:
                            processed_pairs.add(pair_key)
                            pair_block = block_name_to_block[pair_with]
                            group_name = f"sym_pair_{block.name}_{pair_with}"
                            group = [(block, pair_block)]
                            tree_nodes.append(TopologyBStarTree(group_name, group))
                            constrained_blocks.add(block)
                            constrained_blocks.add(pair_block)

                elif sym_type == 'self_symmetric':
                    group_name = f"sym_self_{block.name}"
                    group = [(block, block)]
                    tree_nodes.append(TopologyBStarTree(group_name, group))
                    constrained_blocks.add(block)

        # Add unconstrained blocks
        unconstrained_blocks = [b for b in self.blocks if b not in constrained_blocks]
        for block in unconstrained_blocks:
            tree_nodes.append(TopologyBStarTree(block))

        self.bstar_tree_nodes = tree_nodes

        # Build the main B* tree structure
        if tree_nodes:
            self.rng.shuffle(tree_nodes)
            self.root = tree_nodes[0]
            tree_nodes[0].parent = None

            # Create left-child chain (x_child connections)
            for i in range(len(tree_nodes) - 1):
                tree_nodes[i + 1].parent = tree_nodes[i]
                tree_nodes[i].x_child = tree_nodes[i + 1]
                tree_nodes[i].y_child = None

            tree_nodes[-1].x_child = None
            tree_nodes[-1].y_child = None

            # Build internal tree structures for each node
            for tree_node in tree_nodes:
                if len(tree_node.units) > 1:
                    self.rng.shuffle(tree_node.units)
                    tree_node.root = tree_node.units[0]
                    tree_node.units[0].parent = None

                    # Create y_child chain for symmetry units
                    for j in range(len(tree_node.units) - 1):
                        tree_node.units[j + 1].parent = tree_node.units[j]
                        tree_node.units[j].x_child = None
                        tree_node.units[j].y_child = tree_node.units[j + 1]

                    tree_node.units[-1].x_child = None
                    tree_node.units[-1].y_child = None

        return self.root

    def serialize_tree_to_dict(self, node, node_type="tree_node"):
        """Converts B* tree structure to dictionary format."""
        if not node:
            return None

        node_dict = {
            "name": node.name,
            "type": node_type,
            "x_min": node.x_min,
            "y_min": node.y_min,
            "x_max": node.x_max,
            "y_max": node.y_max
        }

        if hasattr(node, 'units') and node.units:
            node_dict["units"] = []
            for unit in node.units:
                unit_dict = {
                    "l_half": {
                        "name": unit.l_half.name,
                        "width": unit.l_half.width,
                        "height": unit.l_half.height,
                        "x_min": unit.l_half.x_min,
                        "y_min": unit.l_half.y_min,
                        "x_max": unit.l_half.x_max,
                        "y_max": unit.l_half.y_max
                    },
                    "r_half": {
                        "name": unit.r_half.name,
                        "width": unit.r_half.width,
                        "height": unit.r_half.height,
                        "x_min": unit.r_half.x_min,
                        "y_min": unit.r_half.y_min,
                        "x_max": unit.r_half.x_max,
                        "y_max": unit.r_half.y_max
                    }
                }

                if unit.x_child:
                    unit_dict["x_child"] = self.serialize_tree_to_dict(unit.x_child, "symmetry_unit")
                if unit.y_child:
                    unit_dict["y_child"] = self.serialize_tree_to_dict(unit.y_child, "symmetry_unit")

                node_dict["units"].append(unit_dict)

        if hasattr(node, 'x_child') and node.x_child:
            node_dict["x_child"] = self.serialize_tree_to_dict(node.x_child, "tree_node")
        if hasattr(node, 'y_child') and node.y_child:
            node_dict["y_child"] = self.serialize_tree_to_dict(node.y_child, "tree_node")

        return node_dict

    def create_bstar_tree_structure(self):
        """Creates the complete B* tree structure and adds it to processed data."""
        if not self.processed_data:
            return None

        # Extract blocks and build tree
        self.extract_blocks_from_json()
        tree_root = self.build_bstar_tree()

        if not tree_root:
            print("Failed to build B* tree.")
            return None

        # Serialize tree structure
        bstar_tree_dict = self.serialize_tree_to_dict(tree_root)

        # Add B* tree structure to processed data
        self.processed_data["bstar_tree"] = {
            "description": "B* tree structure created from blocks",
            "root": bstar_tree_dict,
            "total_nodes": len(self.bstar_tree_nodes),
            "symmetry_groups": len(self.symmetry_groups),
            "blocks_order": [block.name for block in self.blocks]
        }

        return self.processed_data

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

    def process_json_and_build_tree(self):
        """Main processing method that builds B* tree from loaded JSON."""
        result = self.create_bstar_tree_structure()
        return result

    def main_n8n(self):
        """Main method for n8n integration."""
        self.load_from_n8n()
        self.process_json_and_build_tree()
        self.output_to_n8n()

    def main_file(self, input_filename, output_filename):
        """Main method for file-based processing."""
        if not self.load_json_file(input_filename):
            return False

        if not self.process_json_and_build_tree():
            return False

        return self.save_to_file(output_filename)


def build_bstar_tree_from_files(input_filename, output_filename):
    """
    Convenience function to build B* tree from input file and save to output file.

    Args:
        input_filename (str): Input JSON filename (without extension)
        output_filename (str): Output JSON filename (without extension)

    Returns:
        bool: True if successful, False otherwise
    """
    builder = BStarTreeBuilder()
    input_file = f"{input_filename}.json"
    output_file = f"{output_filename}.json"

    success = builder.main_file(input_file, output_file)

    if success:
        print(f"Successfully created B* tree structure.")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Processed {len(builder.blocks)} blocks")
        print(f"Created {len(builder.bstar_tree_nodes)} tree nodes")
    else:
        print(f"Failed to process files.")

    return success


if __name__ == "__main__":
    # Check if running in n8n mode or file mode
    if len(sys.argv) > 1 and sys.argv[1] == "n8n":
        builder = BStarTreeBuilder()
        builder.main_n8n()
    else:
        # Default file mode for testing
        input_filename = 'loadDevices_in01'
        output_filename = 'loadDevices_out01'
        build_bstar_tree_from_files(input_filename, output_filename)