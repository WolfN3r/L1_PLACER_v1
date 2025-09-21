import json
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from matplotlib.patches import Rectangle, Circle


class VisualizationBlock:
    """Represents a block for visualization with position information."""

    def __init__(self, name):
        self.name = name
        self.width = 0.0
        self.height = 0.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.device_type = ""
        self.current_variant = None
        self.color = "FFFFFF"  # Default white color

    def get_center_x(self):
        return (self.x_min + self.x_max) * 0.5

    def get_center_y(self):
        return (self.y_min + self.y_max) * 0.5

    def get_hex_color(self):
        """Convert hex color to matplotlib color format."""
        if self.color and len(self.color) == 6:
            try:
                return f"#{self.color}"
            except ValueError:
                return "#FFFFFF"  # Default to white if invalid
        return "#FFFFFF"


class VisualizationTreeNode:
    """Represents a B* tree node for visualization."""

    def __init__(self, name):
        self.name = name
        self.units = []
        self.x_child = None
        self.y_child = None
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.color = "FFFFFF"  # Default white color

    def get_hex_color(self):
        """Convert hex color to matplotlib color format."""
        if self.color and len(self.color) == 6:
            try:
                return f"#{self.color}"
            except ValueError:
                return "#FF8C00"  # Default to orange if invalid
        return "#FF8C00"


class VisualizationUnit:
    """Represents a symmetry unit for visualization."""

    def __init__(self, l_half, r_half):
        self.l_half = l_half
        self.r_half = r_half
        self.x_child = None
        self.y_child = None


class PlacementVisualizer:
    """Main class for visualizing placement results."""

    def __init__(self):
        self.original_data = None
        self.blocks = {}  # name -> VisualizationBlock
        self.tree_root = None
        self.nets = []
        self.placement_statistics = {}

    def load_json_file(self, input_filename):
        """Loads a JSON file and stores the data."""
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                self.original_data = json.load(f)
            return self.original_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            self.original_data = None
            return None

    def load_from_n8n(self):
        """Loads JSON data from stdin (for n8n integration)."""
        try:
            self.original_data = json.load(sys.stdin)
            return self.original_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from stdin: {e}", file=sys.stderr)
            self.original_data = None
            return None

    def extract_data_for_visualization(self):
        """Extracts placement data and tree structure for visualization."""
        if not self.original_data or 'bstar_tree' not in self.original_data:
            print("No B* tree data found in JSON.")
            return False

        # Extract placement statistics
        if 'placement_statistics' in self.original_data['bstar_tree']:
            self.placement_statistics = self.original_data['bstar_tree']['placement_statistics']

        # Extract nets information
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

        # Parse B* tree structure and extract block positions
        bstar_tree_data = self.original_data['bstar_tree']
        if 'root' in bstar_tree_data:
            self.tree_root = self.parse_tree_node(bstar_tree_data['root'])

        return True

    def parse_tree_node(self, node_data):
        """Recursively parse B* tree node from JSON data and extract block positions."""
        if not node_data:
            return None

        node = VisualizationTreeNode(node_data['name'])
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

                # Extract l_half block
                l_half_name = l_half_data.get('name', '')
                if l_half_name:
                    if l_half_name not in self.blocks:
                        self.blocks[l_half_name] = VisualizationBlock(l_half_name)

                    l_half_block = self.blocks[l_half_name]
                    l_half_block.width = l_half_data.get('width', 0.0)
                    l_half_block.height = l_half_data.get('height', 0.0)
                    l_half_block.x_min = l_half_data.get('x_min', 0.0)
                    l_half_block.y_min = l_half_data.get('y_min', 0.0)
                    l_half_block.x_max = l_half_data.get('x_max', 0.0)
                    l_half_block.y_max = l_half_data.get('y_max', 0.0)
                    l_half_block.current_variant = l_half_data.get('current_variant')
                    l_half_block.color = l_half_data.get('color', 'FFFFFF')

                # Extract r_half block
                r_half_name = r_half_data.get('name', '')
                if r_half_name:
                    if r_half_name not in self.blocks:
                        self.blocks[r_half_name] = VisualizationBlock(r_half_name)

                    r_half_block = self.blocks[r_half_name]
                    r_half_block.width = r_half_data.get('width', 0.0)
                    r_half_block.height = r_half_data.get('height', 0.0)
                    r_half_block.x_min = r_half_data.get('x_min', 0.0)
                    r_half_block.y_min = r_half_data.get('y_min', 0.0)
                    r_half_block.x_max = r_half_data.get('x_max', 0.0)
                    r_half_block.y_max = r_half_data.get('y_max', 0.0)
                    r_half_block.current_variant = r_half_data.get('current_variant')
                    r_half_block.color = r_half_data.get('color', 'FFFFFF')

                # Create visualization unit
                l_half = self.blocks.get(l_half_name)
                r_half = self.blocks.get(r_half_name)
                if l_half and r_half:
                    unit = VisualizationUnit(l_half, r_half)
                    node.units.append(unit)

        # Extract device types from original blocks data
        if 'blocks' in self.original_data:
            for block_data in self.original_data['blocks']:
                block_name = block_data['name']
                if block_name in self.blocks:
                    self.blocks[block_name].device_type = block_data.get('device_type', '')

        # Parse child nodes
        node.x_child = self.parse_tree_node(node_data.get('x_child'))
        node.y_child = self.parse_tree_node(node_data.get('y_child'))

        return node

    def is_symmetry_node(self, node_name):
        """Check if a tree node represents a symmetry group."""
        return node_name.startswith('sym_pair_') or node_name.startswith('sym_self_')

    def visualize_placement_and_tree(self):
        """Visualizes both block placement and tree structure in one window."""
        if not self.blocks or not self.tree_root:
            print("No data to visualize.")
            return

        fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(16, 8))

        # --- Block placement (left side) ---
        blocks_list = list(self.blocks.values())

        # Calculate bounding box
        x_min = min(block.x_min for block in blocks_list)
        x_max = max(block.x_max for block in blocks_list)
        y_min = min(block.y_min for block in blocks_list)
        y_max = max(block.y_max for block in blocks_list)

        # Draw blocks with their assigned colors
        for block in blocks_list:
            color = block.get_hex_color()
            rect = Rectangle((block.x_min, block.y_min),
                             block.x_max - block.x_min,
                             block.y_max - block.y_min,
                             fill=True, edgecolor='black', facecolor=color, alpha=0.8)
            ax_blocks.add_patch(rect)

            center_x = block.get_center_x()
            center_y = block.get_center_y()

            # Use white text for dark colors, black text for light colors
            text_color = self.get_text_color(color)
            ax_blocks.text(center_x, center_y, block.name, ha='center', va='center',
                           fontsize=10, fontweight='bold', color=text_color)

        # Draw net connections
        if self.nets:
            for net_name, net_blocks in self.nets:
                if len(net_blocks) > 1:
                    centers = [(block.get_center_x(), block.get_center_y()) for block in net_blocks]
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            x0, y0 = centers[i]
                            x1, y1 = centers[j]
                            ax_blocks.plot([x0, x1], [y0, y1], color='red', linewidth=1.5, alpha=0.6, zorder=1)

        ax_blocks.set_xlim(x_min - 1, x_max + 1)
        ax_blocks.set_ylim(y_min - 1, y_max + 1)
        ax_blocks.set_aspect('equal')
        ax_blocks.set_title("Block Placement with Custom Colors", fontsize=14, fontweight='bold')
        ax_blocks.set_xlabel('X Coordinate')
        ax_blocks.set_ylabel('Y Coordinate')

        # --- Tree structure (right side) ---
        def plot_node(node, x, y, depth, step=2.0):
            # Use simple color scheme for tree nodes
            if depth == 1:  # Root node
                color = 'darkgreen'
            elif self.is_symmetry_node(node.name):  # Symmetry nodes
                color = 'blue'
            else:  # Regular tree nodes
                color = 'orange'

            circle = Circle((x, y), 0.2, color=color, ec='black', zorder=2)
            ax_tree.add_patch(circle)

            ax_tree.text(x, y - 0.6, node.name, ha='center', va='top', fontsize=9,
                         zorder=3, color='black', fontweight='bold')

            if hasattr(node, 'x_child') and node.x_child:
                x_l = x + step
                y_l = y
                ax_tree.arrow(x, y, step - 0.2, 0, head_width=0.15, head_length=0.2, fc='green', ec='green',
                              length_includes_head=True)
                plot_node(node.x_child, x_l, y_l, depth + 1, step)

            if hasattr(node, 'y_child') and node.y_child:
                x_r = x
                y_r = y + step
                ax_tree.arrow(x, y, 0, step - 0.2, head_width=0.15, head_length=0.2, fc='blue', ec='blue',
                              length_includes_head=True)
                plot_node(node.y_child, x_r, y_r, depth + 1, step)

        plot_node(self.tree_root, 0, 0, 1)
        ax_tree.set_aspect('equal')
        ax_tree.axis('off')
        ax_tree.set_title("B* Tree Structure with Custom Colors", fontsize=14, fontweight='bold')

        # Add legend for tree
        legend_elements_tree = [
            plt.Line2D([0], [0], color='green', lw=2, label='X-child (horizontal)'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Y-child (vertical)'),
            plt.Circle((0, 0), 0.1, color='darkgreen', label='Root node'),
            plt.Circle((0, 0), 0.1, color='orange', label='Regular nodes'),
            plt.Circle((0, 0), 0.1, color='blue', label='Symmetry nodes')
        ]
        ax_tree.legend(handles=legend_elements_tree, loc='upper right')

        plt.tight_layout()
        plt.show()

    def get_text_color(self, hex_color):
        """Determine if text should be black or white based on background color brightness."""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')

            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            # Calculate brightness (luminance)
            brightness = (r * 0.299 + g * 0.587 + b * 0.114)

            # Return white text for dark backgrounds, black for light
            return 'white' if brightness < 128 else 'black'
        except (ValueError, IndexError):
            return 'black'  # Default to black if color parsing fails

    def print_placement_statistics(self):
        """Prints placement statistics."""
        if self.placement_statistics:
            print("\n" + "=" * 50)
            print("PLACEMENT STATISTICS")
            print("=" * 50)
            print(f"Total blocks: {self.placement_statistics.get('total_blocks', 'N/A')}")
            print(f"Placed blocks: {self.placement_statistics.get('placed_blocks', 'N/A')}")
            print(f"Total block area: {self.placement_statistics.get('total_block_area', 'N/A'):.3f}")

            bbox = self.placement_statistics.get('bounding_box', {})
            print(f"Bounding box: {bbox.get('width', 'N/A'):.2f} x {bbox.get('height', 'N/A'):.2f}")
            print(f"Bounding box area: {bbox.get('area', 'N/A'):.3f}")
            print(f"Utilization: {self.placement_statistics.get('utilization_percent', 'N/A'):.2f}%")
            print(f"Aspect ratio: {self.placement_statistics.get('aspect_ratio', 'N/A'):.3f}")

            device_counts = self.placement_statistics.get('device_type_counts', {})
            if device_counts:
                print("\nDevice type counts:")
                for device_type, count in device_counts.items():
                    print(f"  {device_type}: {count}")

        else:
            print("No placement statistics available.")

    def visualize_all(self):
        """Creates the main visualization with placement and tree."""
        self.print_placement_statistics()
        self.visualize_placement_and_tree()

    def main_local(self, input_filename):
        """Main method for local file processing."""
        input_file = f"{input_filename}.json"

        if not self.load_json_file(input_file):
            print(f"Error: Could not load input file {input_file}")
            return False

        if not self.extract_data_for_visualization():
            print(f"Error: Could not extract data for visualization")
            return False

        print(f"Successfully loaded placement data from: {input_file}")
        self.visualize_all()
        return True

    def main_n8n(self):
        """Main method for n8n integration."""
        if not self.load_from_n8n():
            print("Error: Could not load data from stdin", file=sys.stderr)
            return False

        if not self.extract_data_for_visualization():
            print("Error: Could not extract data for visualization", file=sys.stderr)
            return False

        self.visualize_all()
        return True


def visualize_placement_from_file(input_filename):
    """
    Convenience function to visualize placement from input file.

    Args:
        input_filename (str): Input JSON filename (without extension)

    Returns:
        bool: True if successful, False otherwise
    """
    visualizer = PlacementVisualizer()
    input_file = f"{input_filename}.json"

    success = visualizer.main_local(input_filename)

    if success:
        print(f"Successfully visualized placement from: {input_file}")
    else:
        print(f"Failed to visualize placement from: {input_file}")

    return success


if __name__ == "__main__":
    # n8n integration mode only
    visualizer = PlacementVisualizer()
    visualizer.main_n8n()