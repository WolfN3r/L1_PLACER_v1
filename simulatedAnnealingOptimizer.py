import json
import sys
import copy


class SimulatedAnnealingOptimizer:
    """
    Loader pro načtení JSON s bloky, netlistem a B* stromem.
    Vrací strukturu vhodnou pro další zpracování v tree2.py.
    """

    def __init__(self):
        self.placement_data = None

    def load_json_file(self, input_filename):
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Vyber pouze důležité části pro placement
        self.placement_data = {
            "blocks": data.get("blocks", []),
            "netlist": data.get("netlist", {}),
            "bstar_tree": data.get("bstar_tree", {})
        }
        return self.placement_data

    def swap_x_y_children_in_bstar_tree(self, node):
        """
        Rekurzivně prohodí x_child a y_child ve stromu.
        """
        if not node or not isinstance(node, dict):
            return
        # Swap x_child <-> y_child
        x_child = node.get('x_child')
        y_child = node.get('y_child')
        node['x_child'], node['y_child'] = y_child, x_child
        # Rekurzivně pokračuj do podstromů
        if node['x_child']:
            self.swap_x_y_children_in_bstar_tree(node['x_child'])
        if node['y_child']:
            self.swap_x_y_children_in_bstar_tree(node['y_child'])

    def main_local(self, input_file, output_file, *args, **kwargs):
        # Načti JSON a ulož do self.placement_data
        filename = input_file if input_file.endswith('.json') else input_file + '.json'
        self.load_json_file(filename)
        # Prohoď x_child a y_child v B* stromu
        bstar_tree = self.placement_data.get("bstar_tree", {})
        if "root" in bstar_tree:
            self.swap_x_y_children_in_bstar_tree(bstar_tree["root"])
        # Ulož zpět do output_file pokud je zadán
        if output_file:
            outname = output_file if output_file.endswith('.json') else output_file + '.json'
            with open(outname, 'w', encoding='utf-8') as f:
                json.dump(self.placement_data, f, indent=2, ensure_ascii=False)
        print(json.dumps(self.placement_data, indent=2, ensure_ascii=False))
        return self.placement_data

    def main(self):
        # Pro n8n: načti JSON ze stdin
        data = json.load(sys.stdin)
        self.placement_data = {
            "blocks": data.get("blocks", []),
            "netlist": data.get("netlist", {}),
            "bstar_tree": data.get("bstar_tree", {})
        }
        bstar_tree = self.placement_data.get("bstar_tree", {})
        if "root" in bstar_tree:
            self.swap_x_y_children_in_bstar_tree(bstar_tree["root"])
        print(json.dumps(self.placement_data, indent=2, ensure_ascii=False))
        return self.placement_data


if __name__ == "__main__":
    # Lokální test: načti a vypiš
    optimizer = SimulatedAnnealingOptimizer()
    optimizer.main_local("initPlacement_out01.json", "swappedPlacement_out01.json")
