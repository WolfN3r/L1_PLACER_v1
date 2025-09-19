import json
import sys
import random


class BStarTreeBuilder:
    def __init__(self):
        self.original_data = None
        self.processed_data = None

    def load_json_file(self, input_filename):
        """Loads a JSON file and stores the data in self.original_data."""
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                self.original_data = json.load(f)
            self.processed_data = None  # reset when loading new file
            return self.original_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            self.original_data = None
            self.processed_data = None
            return None

    def save_json_file(self, output_filename):
        """Saves the processed data to a JSON file."""
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

    def create_bstar_tree_structure(self):
        """Creates B* tree structure from blocks and adds it to the data."""
        if not self.original_data or 'blocks' not in self.original_data:
            print("No blocks data found.")
            return None

        # Copy original data
        self.processed_data = self.original_data.copy()

        blocks = self.original_data['blocks']

        # Create B* tree structure
        bstar_tree = self._build_bstar_tree(blocks)

        # Add B* tree structure to the processed data
        self.processed_data['bstar_tree'] = bstar_tree

        return self.processed_data

    def _build_bstar_tree(self, blocks):
        """Build B* tree structure from blocks."""
        if not blocks:
            return None

        # Create tree nodes from blocks
        tree_nodes = []
        for i, block in enumerate(blocks):
            node = {
                "id": i,
                "name": block["name"],
                "block_ref": block["name"],
                "device_type": block.get("device_type", ""),
                "x_child": None,
                "y_child": None,
                "parent": None,
                "position": {
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 0.0,
                    "y_max": 0.0
                },
                "dimensions": {
                    "width": block["variants"][0]["width"] if block.get("variants") else 0.0,
                    "height": block["variants"][0]["height"] if block.get("variants") else 0.0
                },
                "symmetry": block.get("symmetry", {}),
                "is_root": False
            }
            tree_nodes.append(node)

        # Build tree structure - create a left-child right-sibling representation
        # This follows the B* tree structure from tree2.py
        if tree_nodes:
            # Shuffle for random initial tree (as in original tree2.py)
            shuffled_indices = list(range(len(tree_nodes)))
            random.shuffle(shuffled_indices)

            # Set root
            root_idx = shuffled_indices[0]
            tree_nodes[root_idx]["is_root"] = True
            tree_nodes[root_idx]["parent"] = None

            # Build left-child chain (x_child relationships)
            for i in range(len(shuffled_indices) - 1):
                current_idx = shuffled_indices[i]
                next_idx = shuffled_indices[i + 1]

                tree_nodes[current_idx]["x_child"] = next_idx
                tree_nodes[next_idx]["parent"] = current_idx

            # Handle symmetry groups and y_child relationships
            self._handle_symmetry_relationships(tree_nodes, blocks)

        return {
            "nodes": tree_nodes,
            "root": root_idx if tree_nodes else None,
            "structure_type": "bstar_tree",
            "creation_method": "left_child_chain_with_symmetry"
        }

    def _handle_symmetry_relationships(self, tree_nodes, blocks):
        """Handle symmetry constraints in the B* tree structure."""
        # Group blocks by symmetry constraints
        symmetry_groups = {}

        for i, block in enumerate(blocks):
            symmetry_info = block.get("symmetry", {})
            if symmetry_info.get("type") == "pair_symmetric":
                pair_with = symmetry_info.get("pair_with")
                if pair_with:
                    # Find the pair block
                    pair_idx = None
                    for j, other_block in enumerate(blocks):
                        if other_block["name"] == pair_with:
                            pair_idx = j
                            break

                    if pair_idx is not None:
                        # Create symmetry group
                        group_key = f"sym_{min(i, pair_idx)}_{max(i, pair_idx)}"
                        if group_key not in symmetry_groups:
                            symmetry_groups[group_key] = []
                        symmetry_groups[group_key].extend([i, pair_idx])
            elif symmetry_info.get("type") == "self_symmetric":
                # Self-symmetric blocks
                tree_nodes[i]["symmetry_constraint"] = "self_symmetric"

        # Add symmetry group information to tree nodes
        for group_key, group_indices in symmetry_groups.items():
            for idx in group_indices:
                if idx < len(tree_nodes):
                    tree_nodes[idx]["symmetry_group"] = group_key
                    tree_nodes[idx]["symmetry_constraint"] = "pair_symmetric"

    def get_processed_data(self):
        """Return processed data with B* tree structure."""
        return self.processed_data

    def get_original_data(self):
        """Return original data."""
        return self.original_data

    def print_bstar_tree(self):
        """Print B* tree structure in a readable format."""
        if not self.processed_data or 'bstar_tree' not in self.processed_data:
            print("No B* tree structure found.")
            return

        bstar_tree = self.processed_data['bstar_tree']
        nodes = bstar_tree['nodes']
        root_id = bstar_tree['root']

        print(f"B* Tree Structure (Root: {root_id})")
        print("=" * 50)

        for node in nodes:
            print(f"Node {node['id']}: {node['name']}")
            print(f"  Device Type: {node['device_type']}")
            print(f"  Dimensions: {node['dimensions']['width']} x {node['dimensions']['height']}")
            print(f"  X-Child: {node['x_child']}")
            print(f"  Y-Child: {node['y_child']}")
            print(f"  Parent: {node['parent']}")
            print(f"  Is Root: {node['is_root']}")
            if 'symmetry_group' in node:
                print(f"  Symmetry Group: {node['symmetry_group']}")
            print()

    #######################################################################################
    # n8n integration methods
    #######################################################################################
    def load_from_n8n(self):
        """Loads JSON data from stdin (for n8n integration)."""
        try:
            self.original_data = json.load(sys.stdin)
            self.processed_data = None
            return self.original_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from stdin: {e}", file=sys.stderr)
            self.original_data = None
            self.processed_data = None
            return None

    def output_to_n8n(self):
        """Outputs processed data to stdout (for n8n integration)."""
        if self.processed_data is not None:
            print(json.dumps(self.processed_data))
        else:
            print("No processed data available.", file=sys.stderr)

    def main(self):
        """Main method for n8n integration."""
        self.load_from_n8n()
        self.create_bstar_tree_structure()
        self.output_to_n8n()

    #######################################################################################
    # Local file processing methods
    #######################################################################################
    def process_local_files(self, input_filename, output_filename):
        """Process local JSON files - load, create B* tree, and save."""
        # Load input file
        if not self.load_json_file(input_filename):
            return False

        # Create B* tree structure
        if not self.create_bstar_tree_structure():
            print("Failed to create B* tree structure.")
            return False

        # Save output file
        if not self.save_json_file(output_filename):
            return False

        print(f"Successfully processed {input_filename} -> {output_filename}")
        return True


#######################################################################################
# Main function for n8n integration
#######################################################################################
if __name__ == "__main__":
    builder = BStarTreeBuilder()
    builder.main()