import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import math


################################################################################
class hard_block:
    " It represents l_half or r_half in general case."

    def __init__(self, name):
        self.name = name
        self.merge_device_names = []
        self.patterns = []
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.width = 0.0
        self.height = 0.0
        self.column_multiple = 1
        self.row_multiple = 1
        self.variants = []

    def get_width(self):
        return self.x_max - self.x_min if self.x_max > self.x_min else self.width

    def get_height(self):
        return self.y_max - self.y_min if self.y_max > self.y_min else self.height

    def get_x_min(self):
        return self.x_min

    def get_x_max(self):
        return self.x_max

    def get_y_min(self):
        return self.y_min

    def get_y_max(self):
        return self.y_max

    def SetW(self, w):
        self.width = w

    def SetH(self, h):
        self.height = h

    def GetCenterX(self):
        return (self.x_min + self.x_max) * 0.5

    def GetCenterY(self):
        return (self.y_min + self.y_max) * 0.5

    # def GetColumnMultiple
    # def GetColumnMultiple(self):
    #     return self.column_multiple
    #
    # def GetRowMultiple(self):
    #     return self.row_multiple


class SymmetryUnit:
    def __init__(self, r_half, l_half):
        self.r_half = r_half
        self.l_half = l_half
        self.x_child = None
        self.y_child = None
        self.parent = None
        self.l_hint = None
        self.r_hint = None
        if r_half != l_half:
            r_half.patterns.sort()
            l_half.patterns.sort()
            # assert l_half.patterns and r_half.patterns  # ZAKOMENTUJTE TUTO ŘÁDKU
            # assert l_half.patterns == r_half.patterns   # ZAKOMENTUJTE TUTO ŘÁDKU
            l_half.SetW(l_half.patterns[0].width if l_half.patterns else l_half.width)
            l_half.SetH(l_half.patterns[0].height if l_half.patterns else l_half.height)
            r_half.SetW(r_half.patterns[0].width if r_half.patterns else r_half.width)
            r_half.SetH(r_half.patterns[0].height if r_half.patterns else r_half.height)


class topology_BStarTree:
    def __init__(self, block_or_group, group=None):
        if group is not None:
            self.name = str(block_or_group)
            self.units = [SymmetryUnit(b1, b2) for b1, b2 in group]
        elif isinstance(block_or_group, hard_block):
            self.name = block_or_group.name
            self.units = [SymmetryUnit(block_or_group, block_or_group)]
        else:
            self.name = str(block_or_group)
            self.units = [SymmetryUnit(b1, b2) for b1, b2 in block_or_group]
        self.root = self.units[0]
        self.x_child = None
        self.y_child = None
        self.parent = None
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0


class Loader:
    ############################################################################
    def __init__(self):
        self.block_name_to_block = {}
        self.net_name_to_blocks = {}
        self.sym_group_name_to_sym_group = {}
        self.hard_blocks = set()

    ############################################################################
    def Load(self, NETLIST_FILE, SYMMETRY_FILE, BLOCK_FILE):
        if not self.LoadBlockFile(BLOCK_FILE):
            print(f"Failed to open file: {BLOCK_FILE}")
            return False
        if not self.LoadNetlistFile(NETLIST_FILE):
            print(f"Failed to open file: {NETLIST_FILE}")
            return False
        if not self.LoadSymmetryConstraintFile(SYMMETRY_FILE):
            print(f"Failed to open file: {SYMMETRY_FILE}")
            return False
        return True

    ############################################################################
    def LoadBlockFile(self, filepath):
        try:
            with open(filepath) as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    tokens = line.split()
                    name = tokens[0]
                    block = hard_block(name)
                    merge_names = []
                    variants = []
                    i = 1
                    while i < len(tokens) and not tokens[i].startswith('('):
                        merge_names.append(tokens[i])
                        i += 1
                    while i < len(tokens):
                        if tokens[i].startswith('('):
                            param_tokens = []
                            param_tokens.append(tokens[i][1:])
                            i += 1
                            while i < len(tokens) and not tokens[i].endswith(')'):
                                param_tokens.append(tokens[i])
                                i += 1
                            if i < len(tokens):
                                param_tokens.append(tokens[i][:-1])
                            param_vals = param_tokens
                            if len(param_vals) >= 4:
                                variants.append((
                                    float(param_vals[0]),
                                    float(param_vals[1]),
                                    int(param_vals[2]),
                                    int(param_vals[3])
                                ))
                        i += 1
                    if not variants:
                        continue  # ignorovat blok bez parametrů
                    block.width = variants[0][0]
                    block.height = variants[0][1]
                    block.row_multiple = variants[0][2]
                    block.column_multiple = variants[0][3]
                    block.variants = variants[0:]  # dba changed copilot data in order to have all versions
                    block.merge_device_names = merge_names
                    self.block_name_to_block[block.name] = block
                    for merge_name in merge_names:
                        self.block_name_to_block[merge_name] = block
                    self.hard_blocks.add(block)
            return True
        except Exception as e:
            print(f"Chyba při načítání bloků: {e}")
            return False

    ############################################################################
    def LoadNetlistFile(self, filepath):
        try:
            with open(filepath) as file:
                device_name_to_block = {}
                for name, block in self.block_name_to_block.items():
                    if not block.merge_device_names:
                        device_name_to_block[name] = block
                    else:
                        for merge_name in block.merge_device_names:
                            device_name_to_block[merge_name] = block
                for line in file:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    device_name = tokens[0]
                    if device_name.startswith('M') and len(tokens) >= 4:
                        d, g, s = tokens[1:4]
                        d = d.lower()
                        g = g.lower()
                        s = s.lower()
                        block = device_name_to_block.get(device_name)
                        if block:
                            self.net_name_to_blocks.setdefault(d, set()).add(block)
                            self.net_name_to_blocks.setdefault(g, set()).add(block)
                            self.net_name_to_blocks.setdefault(s, set()).add(block)
            return True
        except Exception as e:
            print(f"Chyba při načítání netlistu: {e}")
            return False

    ############################################################################
    def LoadSymmetryConstraintFile(self, filepath):
        try:
            with open(filepath) as file:
                for line in file:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    group_name = tokens[0]
                    block_name1 = tokens[1]
                    block1 = self.block_name_to_block.get(block_name1)
                    if not block1:
                        continue
                    if len(tokens) > 2:
                        block_name2 = tokens[2]
                        block2 = self.block_name_to_block.get(block_name2)
                        if not block2:
                            continue
                    else:
                        block2 = block1
                    self.sym_group_name_to_sym_group.setdefault(group_name, set()).add((block1, block2))
            return True
        except Exception as e:
            print(f"Chyba při načítání symetrie: {e}")
            return False

    ############################################################################
    def get_nets(self):
        nets = []
        for name, blocks in self.net_name_to_blocks.items():
            nets.append((name, list(blocks)))
        return nets

    ############################################################################
    def get_virtual_hier(self):
        constrained_blocks = set()
        for group in self.sym_group_name_to_sym_group.values():
            for block1, block2 in group:
                constrained_blocks.add(block1)
                if block1 != block2:
                    constrained_blocks.add(block2)

        unconstrained_blocks = self.hard_blocks - constrained_blocks

        nodes = []
        for block in unconstrained_blocks:
            nodes.append(topology_BStarTree(block))

        for group_name, group in self.sym_group_name_to_sym_group.items():
            nodes.append(topology_BStarTree(group_name, group))
        return nodes


################################################################################
class dba_perturbator_class:
    def __init__(self, hbtree):
        self.hbtree = hbtree
        self.l_virtual_hier_w_hard_blocks = hbtree.l_virtual_hier_w_hard_blocks
        self.hier_trees_ = hbtree.l_virtual_hier_w_symmetry
        # self.rebuildable_trees_ = [t for t in self.hier_trees_ if len(t.units) > 1]
        # self.resizeable_units_ = [u for t in self.hier_trees_ for u in t.units if len(u.r_half.patterns) > 1]
        # self.flipable_units_ = [u for t in self.hier_trees_ for u in t.units if u.l_half != u.r_half]

        self.rebuildable_trees_ = []
        self.resizeable_units_ = []
        self.flipable_units_ = []

        # collect candidates
        for hier_tree in self.hier_trees_:
            if len(hier_tree.units) > 1:
                self.rebuildable_trees_.append(hier_tree)

        for hier_tree in self.hier_trees_:
            for unit in hier_tree.units:
                if unit.l_half != unit.r_half:
                    self.flipable_units_.append(unit)
                if len(unit.r_half.patterns) > 1:
                    self.resizeable_units_.append(unit)

        # perturbation functions
        self.perturb_funcs = [
            self.random_resize_symmetry_unit,
            # self.random_swap_asfbstar_tree,
            self.random_swap_asfbstar_tree_child,
            self.random_flip_symmetry_unit,
            self.random_rebuild_asfbstar_tree,
            # self.random_transplant_asfbstar_tree
        ]
        self.rng = random.Random()

    def __call__(self):
        if not self.perturb_funcs:
            return

        # The random variant choice should respect symmetry request.
        # In case of symmetry, only one variant in unit.l_half and
        # unit.r_half should be used It could be fixed by post-procedure
        # that for instance in case of symmetry the used variant will be
        # the one used in unit.l_half and that will be copy to unit.r_half

        # Nejprve nastav varianty pro všechny symmetry units
        for tree in self.hier_trees_:
            for unit in tree.units:
                if unit.l_half == unit.r_half:
                    block = unit.l_half
                    if hasattr(block, "variants") and block.variants:
                        variant = self.rng.choice(block.variants)
                        block.current_variant = variant
                        block.width = variant[0]
                        block.height = variant[1]
                        block.row_multiple = variant[2]
                        block.column_multiple = variant[3]
                        block.x_min = 0.0
                        block.y_min = 0.0
                        block.x_max = block.width
                        block.y_max = block.height
                else:
                    # Symetrie: vyber jeden variant a použij pro oba bloky
                    block_l = unit.l_half
                    block_r = unit.r_half
                    if hasattr(block_l, "variants") and block_l.variants:
                        variant = self.rng.choice(block_l.variants)
                        for block in (block_l, block_r):
                            block.current_variant = variant
                            block.width = variant[0]
                            block.height = variant[1]
                            block.row_multiple = variant[2]
                            block.column_multiple = variant[3]
                            block.x_min = 0.0
                            block.y_min = 0.0
                            block.x_max = block.width
                            block.y_max = block.height

        # equivalent of std::discrete_distribution
        num_perturb = self.rng.randint(1, len(self.perturb_funcs))
        for _ in range(num_perturb):
            func = self.rng.choice(self.perturb_funcs)
            func()

    def random_resize_symmetry_unit(self):
        if not self.resizeable_units_:
            return
        unit = self.rng.choice(self.resizeable_units_)
        pattern = self.rng.choice(unit.r_half.patterns)
        unit.r_half.SetW(pattern.width)
        unit.r_half.SetH(pattern.height)
        unit.l_half.SetW(pattern.width)
        unit.l_half.SetH(pattern.height)

    def random_swap_asfbstar_tree_child(self):
        if not self.hier_trees_:
            return
        t = self.rng.choice(self.hier_trees_)
        t.x_child, t.y_child = t.y_child, t.x_child

    def random_flip_symmetry_unit(self):
        if not self.flipable_units_:
            return
        unit = self.rng.choice(self.flipable_units_)
        unit.x_child, unit.y_child = unit.y_child, unit.x_child

    def random_rebuild_asfbstar_tree(self):
        if not self.rebuildable_trees_:
            return
        hier_tree = self.rng.choice(self.rebuildable_trees_)
        units = hier_tree.units
        self.rng.shuffle(units)
        hier_tree.root = units[0]
        units[0].parent = None
        for i in range(len(units) - 1):
            units[i].x_child = None
            units[i].y_child = units[i + 1]
        units[-1].x_child = None
        units[-1].y_child = None
        hier_tree.units = units

    def get_rand_child(self, tree):
        if tree.x_child and tree.y_child:
            return self.rng.choice([tree.x_child, tree.y_child])
        if tree.x_child:
            return tree.x_child
        if tree.y_child:
            return tree.y_child
        return None


################################################################################
class dba_top_tree:
    def __init__(self):
        self.l_nets = []
        self.l_virtual_hier_w_hard_blocks = []
        self.l_virtual_hier_w_symmetry = []
        self.rng = random.Random()
        self.root = None
        self.units = []
        self.x_min = 0.0
        self.x_max = 0.0
        self.y_min = 0.0
        self.y_max = 0.0

    def dba_load(self, NETLIST_FILE, SYMMETRY_FILE, BLOCK_FILE):
        loader = Loader()
        if not loader.Load(NETLIST_FILE, SYMMETRY_FILE, BLOCK_FILE):
            return False
        self.l_nets = loader.get_nets()
        self.l_virtual_hier_w_hard_blocks = loader.hard_blocks
        self.l_virtual_hier_w_symmetry = loader.get_virtual_hier()
        self.BuildTree()  # it creates random B*tree
        visualize_tree2(self.root)
        # visualize_tree(self.root)
        # dba_top_tree.PackorSquare(self)()  # it places blocks according to B*tree
        dba_top_tree.place(self, self.root)  # it places blocks according to B*tree
        # visualize_block_positions(self.l_virtual_hier_w_hard_blocks)
        visualize_tree_and_blocks4(self.root, self.l_virtual_hier_w_hard_blocks, self.l_nets)
        return True

    def BuildTree(self):
        # Construct the left list B*-tree
        # Construct TOP level hier. B*-tree (left child only) due to symmetry
        random.shuffle(self.l_virtual_hier_w_symmetry)
        self.root = self.l_virtual_hier_w_symmetry[0]
        self.l_virtual_hier_w_symmetry[0].parent = None
        for i in range(len(self.l_virtual_hier_w_symmetry) - 1):
            self.l_virtual_hier_w_symmetry[i + 1].parent = self.l_virtual_hier_w_symmetry[i]
            self.l_virtual_hier_w_symmetry[i].x_child = self.l_virtual_hier_w_symmetry[i + 1]
            self.l_virtual_hier_w_symmetry[i].y_child = None
        self.l_virtual_hier_w_symmetry[-1].x_child = None
        self.l_virtual_hier_w_symmetry[-1].y_child = None

        # Construct the right list ASF B*-tree for each unit due to symmetry
        # Construct sub-TOP (TOP-1) level hier
        # It make sense in case of symmetry group
        # Tt is recorded into the y_child,
        # because the y_child is placed in vertical way, the x_child in horizontal way
        for block_tree in self.l_virtual_hier_w_symmetry:
            units = block_tree.units
            random.shuffle(units)
            block_tree.root = units[0]
            units[0].parent = None
            for i in range(len(units) - 1):
                units[i + 1].parent = units[i]
                units[i].x_child = None
                units[i].y_child = units[i + 1]
            units[-1].x_child = None
            units[-1].y_child = None

    ############################################################################
    def place(self, root, x_offset=0.0, y_offset=0.0):
        """Spustí hierarchické umisťování od zadaného uzlu."""
        self.reset_block_positions()
        blocks = []
        blocks += self.place_units(root.units, x_offset, y_offset)
        last_bbox = self.get_last_bbox(blocks)
        if root.x_child:
            # Umísti x_child vpravo od posledního bloku
            blocks += self.place_x_child(root.x_child, last_bbox['x_max'], last_bbox['y_min'])
        if root.y_child:
            # Umísti y_child nad poslední blok
            blocks += self.place_y_child(root.y_child, last_bbox['x_min'], last_bbox['y_max'])
        self.update_bbox(root, blocks)
        return blocks

    def place_units(self, units, x_offset, y_offset):
        """Umístí jednotky podle symetrie a počtu."""
        bbox = []
        curr_y = y_offset
        # Najdi maximální šířku levých bloků pro zarovnání na osu
        max_l_width = max((u.l_half.get_width() for u in units), default=0.0)
        min_r_x = x_offset + max_l_width
        for unit in units:
            if unit.r_half == unit.l_half:
                block = unit.r_half
                block.x_min = x_offset
                block.y_min = curr_y
                block.x_max = x_offset + block.get_width()
                block.y_max = curr_y + block.get_height()
                bbox.append(block)
                curr_y += block.get_height()
            else:
                # Zarovnat všechny bloky na osu symetrie
                l = unit.l_half
                r = unit.r_half
                # Osa symetrie je v min_r_x
                l.x_max = min_r_x
                # l.x_min = l.x_max - l.get_width()
                l.x_min = l.x_max - l.width
                l.y_min = curr_y
                # l.y_max = curr_y + l.get_height()
                l.y_max = curr_y + l.height

                r.x_min = min_r_x
                # r.x_max = min_r_x + r.get_width()
                r.x_max = min_r_x + r.width
                r.y_min = curr_y
                # r.y_max = curr_y + r.get_height()
                r.y_max = curr_y + r.height
                bbox.extend([l, r])
                # curr_y += max(l.get_height(), r.get_height())
                curr_y += max(l.height, r.height)
        return bbox

    def place_x_child(self, root, x_offset, y_offset):
        """Umístí x_child vpravo od parenta, se stejným y_min."""
        blocks = []
        blocks += self.place_units(root.units, x_offset, y_offset)
        last_bbox = self.get_last_bbox(blocks)
        if root.x_child:
            blocks += self.place_x_child(root.x_child, last_bbox['x_max'], last_bbox['y_min'])
        if root.y_child:
            blocks += self.place_y_child(root.y_child, last_bbox['x_min'], last_bbox['y_max'])
        return blocks

    def place_y_child(self, root, x_offset, y_offset):
        """Umístí y_child nad parenta, se stejným x_min."""
        blocks = []
        blocks += self.place_units(root.units, x_offset, y_offset)
        last_bbox = self.get_last_bbox(blocks)
        if root.x_child:
            blocks += self.place_x_child(root.x_child, last_bbox['x_max'], last_bbox['y_min'])
        if root.y_child:
            blocks += self.place_y_child(root.y_child, last_bbox['x_min'], last_bbox['y_max'])
        return blocks

    def get_last_bbox(self, blocks):
        """Vrátí bbox posledního bloku (pro navazující umístění)."""
        if not blocks:
            return {'x_min': 0.0, 'x_max': 0.0, 'y_min': 0.0, 'y_max': 0.0}
        x_min = min(b.x_min for b in blocks)
        x_max = max(b.x_max for b in blocks)
        y_min = min(b.y_min for b in blocks)
        y_max = max(b.y_max for b in blocks)
        return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

    def update_bbox(self, node, blocks):
        """Aktualizuje bbox rootu/top úrovně s přesností na tři desetinná místa."""
        if blocks:
            self.x_min = round(min(b.x_min for b in blocks), 3)
            self.y_min = round(min(b.y_min for b in blocks), 3)
            self.x_max = round(max(b.x_max for b in blocks), 3)
            self.y_max = round(max(b.y_max for b in blocks), 3)

    ############################################################################
    def Dump(self, output_file_path):
        try:
            with open(output_file_path, "w") as file:
                file.write(f"{self.get_total_hpwl():.6f}\n")
                file.write(f"{self.get_area():.6f}\n")
                file.write(f"{self.get_width():.6f} {self.get_height():.6f}\n")
                for block in self.l_virtual_hier_w_hard_blocks:
                    file.write(f"{block.name} ")
                    for name in block.merge_device_names:
                        file.write(f"{name} ")
                    file.write(
                        f"{block.get_x_min():.6f} {block.get_y_min():.6f} ({block.get_width():.6f} {block.get_height():.6f} {block.GetColumnMultiple()} {block.GetRowMultiple()})\n")
            return True
        except Exception:
            return False

    def get_width(self):
        return self.x_max - self.x_min

    def get_height(self):
        return self.y_max - self.y_min

    def get_x_min(self):
        return self.x_min

    def get_x_max(self):
        return self.x_max

    def get_y_min(self):
        return self.y_min

    def get_y_max(self):
        return self.y_max

    def GetColumnMultiple(self):
        return self.column_multiple

    def GetRowMultiple(self):
        return self.row_multiple

    def Clear(self):
        self.l_virtual_hier_w_symmetry.clear()
        self.units.clear()
        self.l_virtual_hier_w_hard_blocks.clear()
        self.l_nets.clear()
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.root = None

    def get_area(self):
        return self.get_width() * self.get_height()

    def get_total_hpwl(self):
        total_hpwl = 0.0
        for _, blocks in self.l_nets:
            x0 = x1 = blocks[0].GetCenterX()
            y0 = y1 = blocks[0].GetCenterY()
            for block in blocks[1:]:
                xc = block.GetCenterX()
                yc = block.GetCenterY()
                x0 = min(x0, xc)
                x1 = max(x1, xc)
                y0 = min(y0, yc)
                y1 = max(y1, yc)
            w = x1 - x0
            h = y1 - y0
            total_hpwl += w + h
        return total_hpwl

    def set_x_min(self, x):
        self.x_min = x
        return self

    def set_y_min(self, y):
        self.y_min = y
        return self

    def set_x_max(self, x):
        self.x_max = x
        return self

    def set_y_max(self, y):
        self.y_max = y
        return self

    def save_placement(self):
        # Uloží aktuální pozice bloků
        return [(block.name, block.x_min, block.y_min, block.x_max, block.y_max) for block in
                self.l_virtual_hier_w_hard_blocks]

    def load_placement(self, placement):
        # Obnoví pozice bloků ze záznamu
        name_to_block = {block.name: block for block in self.l_virtual_hier_w_hard_blocks}
        for name, x_min, y_min, x_max, y_max in placement:
            block = name_to_block[name]
            block.x_min = x_min
            block.y_min = y_min
            block.x_max = x_max
            block.y_max = y_max

    def save_tree_state(self):
        # Uloží root a pořadí stromů (deep copy není nutný, pokud se nemění reference uvnitř)
        # return (self.root, list(self.l_virtual_hier_w_symmetry))
        return copy.deepcopy(self)

    def load_tree_state(self, state):
        self.root = state.root
        self.l_virtual_hier_w_symmetry = state.l_virtual_hier_w_symmetry
        self.l_virtual_hier_w_hard_blocks = state.l_virtual_hier_w_hard_blocks
        self.l_nets = state.l_nets
        self.x_min = state.x_min
        self.x_max = state.x_max
        self.y_min = state.y_min
        self.y_max = state.y_max
        self.units = state.units

    def reset_block_positions(self):
        for block in self.l_virtual_hier_w_hard_blocks:
            block.x_min = 0.0
            block.x_max = 0.0
            block.y_min = 0.0
            block.y_max = 0.0

    def optimize(self, cost_fn, timer, patience=10e6):
        placement = self.save_placement()
        tree_state = self.save_tree_state()
        dba_perturbator = dba_perturbator_class(self)
        cost = cost_fn(self)
        reject_count = 0
        j = 1
        while not timer.is_timeout() and reject_count < patience:
            # while reject_count < 2:
            self.reset_block_positions()
            # visualize_tree2(self.root)
            dba_perturbator()
            # visualize_tree2(self.root)
            dba_top_tree.place(self, self.root)
            # visualize_tree_and_blocks2(self.root, self.l_virtual_hier_w_hard_blocks)
            new_cost = cost_fn(self)
            if new_cost < cost:
                cost = new_cost
                # placement = self.save_placement()
                tree_state = self.save_tree_state()
                reject_count = 0
            else:
                # self.load_placement(placement)
                # self.load_tree_state(tree_state)
                reject_count += 1
            j = j + 1
            # print(f"run: {j:03} -> Cost={cost_func(self):.6f}")

        # dba_packor(self)()  # Znovu umístí bloky podle stromu !!! added by dba
        # dba_top_tree.place(self, self.root) # Znovu umístí bloky podle stromu !!! added by dba

        # self.load_placement(placement)
        self.load_tree_state(tree_state)
        # print(f"run: {j:03} -> Cost tree_state={cost_func(tree_state[0]):.6f}")
        print(f"run: {j:03} -> Cost={cost_func(self):.6f}")
        # visualize_tree_and_blocks2(self.root, self.l_virtual_hier_w_hard_blocks)
        # dba_top_tree.place(self, tree_state[0])

        return tree_state


################################################################################
def visualize_tree2(root):
    fig, ax_tree = plt.subplots(figsize=(12, 6))

    # --- Strom vpravo ---
    def plot_node(node, x, y, depth, step=2.0, color='orange'):
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', zorder=2)
        ax_tree.add_patch(circle)
        ax_tree.text(x - 0.4, y - 0.4, node.name, ha='center', va='bottom', fontsize=10, zorder=3)
        # x_child: zelená šipka doprava
        if hasattr(node, 'x_child') and node.x_child:
            x_l = x + step
            y_l = y
            ax_tree.arrow(x, y, step, 0, head_width=0.15, head_length=0.5, fc='green', ec='green',
                          length_includes_head=True)
            plot_node(node.x_child, x_l, y_l, depth + 1, step)
        # y_child: modrá šipka nahoru
        if hasattr(node, 'y_child') and node.y_child:
            x_r = x
            y_r = y + step
            ax_tree.arrow(x, y, 0, step, head_width=0.15, head_length=0.5, fc='blue', ec='blue',
                          length_includes_head=True)
            plot_node(node.y_child, x_r, y_r, depth + 1, step)

    # First point with different color from the others
    plot_node(root, 0, 0, 1, color='darkgreen')
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()


def visualize_tree_and_blocks(root, blocks):
    fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Bloky vlevo ---
    x_min = min(block.x_min for block in blocks)
    x_max = max(block.x_min + block.width for block in blocks)
    y_min = min(block.y_min for block in blocks)
    y_max = max(block.y_min + block.height for block in blocks)
    for block in blocks:
        x = block.x_min
        y = block.y_min
        w = block.width
        h = block.height
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
        ax_blocks.add_patch(rect)
        ax_blocks.text(x + w / 2, y + h / 2, block.name, ha='center', va='center', fontsize=8)
    ax_blocks.set_xlim(x_min, x_max)
    ax_blocks.set_ylim(y_min, y_max)
    ax_blocks.set_aspect('equal')
    ax_blocks.set_title("Blocks positions")
    ax_blocks.set_xlabel('X')
    ax_blocks.set_ylabel('Y')

    # --- Strom vpravo ---
    arrow_length_init = 2.0
    angle_left = -40
    angle_right = 40

    def plot_node(node, x, y, depth, arrow_length, label_side="right"):
        circle = plt.Circle((x, y), 0.2, color='orange', ec='black', zorder=2)
        ax_tree.add_patch(circle)
        if label_side == "left":
            ax_tree.text(x + 0.3, y, node.name, ha='left', va='center', fontsize=8, zorder=3)
        else:
            ax_tree.text(x - 0.3, y, node.name, ha='right', va='center', fontsize=8, zorder=3)
        if hasattr(node, 'x_child') and node.x_child:
            dx = arrow_length * math.cos(math.radians(angle_left)) * 0.9
            dy = -arrow_length * math.sin(math.radians(angle_left)) * 0.9
            x_l = x + dx
            y_l = y - abs(dy)
            ax_tree.arrow(x, y, dx, -abs(dy), head_width=0.15, head_length=0.2, fc='green', ec='green',
                          length_includes_head=True)
            plot_node(node.x_child, x_l, y_l, depth + 1, arrow_length * 0.9, label_side="left")
        if hasattr(node, 'y_child') and node.y_child:
            dx = -arrow_length * math.cos(math.radians(angle_right)) * 0.9
            dy = -arrow_length * math.sin(math.radians(angle_right)) * 0.9
            x_r = x + dx
            y_r = y - abs(dy)
            ax_tree.arrow(x, y, dx, -abs(dy), head_width=0.15, head_length=0.2, fc='blue', ec='blue',
                          length_includes_head=True)
            plot_node(node.y_child, x_r, y_r, depth + 1, arrow_length * 0.9, label_side="right")

    plot_node(root, 0, 0, 1, arrow_length_init, label_side="right")
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()


def visualize_tree_and_blocks2(root, blocks, nets):
    fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Bloky vlevo ---
    x_min = min(block.x_min for block in blocks)
    x_max = max(block.x_min + block.width for block in blocks)
    y_min = min(block.y_min for block in blocks)
    y_max = max(block.y_min + block.height for block in blocks)
    for block in blocks:
        x = block.x_min
        y = block.y_min
        w = block.width
        h = block.height
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
        ax_blocks.add_patch(rect)
        ax_blocks.text(x + w / 2, y + h / 2, block.name, ha='center', va='center', fontsize=8)
    ax_blocks.set_xlim(x_min, x_max)
    ax_blocks.set_ylim(y_min, y_max)
    ax_blocks.set_aspect('equal')
    ax_blocks.set_title("Blocks positions")
    ax_blocks.set_xlabel('X')
    ax_blocks.set_ylabel('Y')

    # vykreslení propojení
    # for net_name, net_blocks in nets:
    #     centers = [(b.x_min + b.width/2, b.y_min + b.height/2) for b in net_blocks]
    #     for i in range(len(centers)):
    #         for j in range(i + 1, len(centers)):
    #             x0, y0 = centers[i]
    #             x1, y1 = centers[j]
    #             ax_blocks.plot([x0, x1], [y0, y1], color='green', linewidth=1, alpha=0.6)

    # --- Strom vpravo ---
    def plot_node(node, x, y, depth, step=2.0, color='orange'):
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', zorder=2)
        ax_tree.add_patch(circle)
        ax_tree.text(x - 0.4, y - 0.4, node.name, ha='center', va='bottom', fontsize=10, zorder=3)
        # x_child: zelená šipka doprava
        if hasattr(node, 'x_child') and node.x_child:
            x_l = x + step
            y_l = y
            ax_tree.arrow(x, y, step, 0, head_width=0.15, head_length=0.5, fc='green', ec='green',
                          length_includes_head=True)
            plot_node(node.x_child, x_l, y_l, depth + 1, step)
        # y_child: modrá šipka nahoru
        if hasattr(node, 'y_child') and node.y_child:
            x_r = x
            y_r = y + step
            ax_tree.arrow(x, y, 0, step, head_width=0.15, head_length=0.5, fc='blue', ec='blue',
                          length_includes_head=True)
            plot_node(node.y_child, x_r, y_r, depth + 1, step)

    # First point with different color from the others
    plot_node(root, 0, 0, 1, color='darkgreen')
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()


def visualize_tree_and_blocks3(root, blocks):
    import matplotlib.pyplot as plt

    # --- Přiřazení barev skupinám podle SymmetryUnit ---
    block_to_group = {}
    group_to_blocks = {}
    group_id = 0

    # Pokud je k dispozici strom, použijeme jeho SymmetryUnit pro skupiny
    if tree is not None and hasattr(tree, "l_virtual_hier_w_symmetry"):
        for btree in tree.l_virtual_hier_w_symmetry:
            for unit in btree.units:
                group = group_id
                for block in (unit.l_half, unit.r_half):
                    block_to_group[block] = group
                    group_to_blocks.setdefault(group, []).append(block)
                group_id += 1
    # fallback: každý blok vlastní skupina
    else:
        for block in blocks:
            block_to_group[block] = group_id
            group_to_blocks.setdefault(group_id, []).append(block)
            group_id += 1

    # Přiřaď barvy skupinám
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    group_colors = {}
    for i, group in enumerate(group_to_blocks):
        group_colors[group] = color_list[i % len(color_list)]

    # --- Bloky vlevo ---
    fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(12, 6))
    x_min = min(block.x_min for block in blocks)
    x_max = max(block.x_min + block.width for block in blocks)
    y_min = min(block.y_min for block in blocks)
    y_max = max(block.y_min + block.height for block in blocks)
    for block in blocks:
        x = block.x_min
        y = block.y_min
        w = block.width
        h = block.height
        color = group_colors[block_to_group[block]]
        rect = plt.Rectangle((x, y), w, h, fill=True, edgecolor='black', facecolor=color, alpha=0.6)
        ax_blocks.add_patch(rect)
        ax_blocks.text(x + w / 2, y + h / 2, block.name, ha='center', va='center', fontsize=8)
    ax_blocks.set_xlim(x_min, x_max)
    ax_blocks.set_ylim(y_min, y_max)
    ax_blocks.set_aspect('equal')
    ax_blocks.set_title("Blocks positions")
    ax_blocks.set_xlabel('X')
    ax_blocks.set_ylabel('Y')

    # --- Strom vpravo (beze změny) ---
    def plot_node(node, x, y, depth, step=2.0, color='orange'):
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', zorder=2)
        ax_tree.add_patch(circle)
        ax_tree.text(x - 0.4, y - 0.4, node.name, ha='center', va='bottom', fontsize=10, zorder=3)
        if hasattr(node, 'x_child') and node.x_child:
            x_l = x + step
            y_l = y
            ax_tree.arrow(x, y, step, 0, head_width=0.15, head_length=0.5, fc='green', ec='green',
                          length_includes_head=True)
            plot_node(node.x_child, x_l, y_l, depth + 1, step)
        if hasattr(node, 'y_child') and node.y_child:
            x_r = x
            y_r = y + step
            ax_tree.arrow(x, y, 0, step, head_width=0.15, head_length=0.5, fc='blue', ec='blue',
                          length_includes_head=True)
            plot_node(node.y_child, x_r, y_r, depth + 1, step)

    plot_node(root, 0, 0, 1, color='darkgreen')
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()


def visualize_tree_and_blocks4(root, blocks, nets):
    import matplotlib.pyplot as plt

    # --- Přiřazení barev skupinám podle SymmetryUnit ---
    block_to_group = {}
    group_to_blocks = {}
    group_id = 0

    # Pokud je k dispozici strom, použijeme jeho SymmetryUnit pro skupiny
    if tree is not None and hasattr(tree, "l_virtual_hier_w_symmetry"):
        for btree in tree.l_virtual_hier_w_symmetry:
            for unit in btree.units:
                group = group_id
                for block in (unit.l_half, unit.r_half):
                    block_to_group[block] = group
                    group_to_blocks.setdefault(group, []).append(block)
                group_id += 1
    # fallback: každý blok vlastní skupina
    else:
        for block in blocks:
            block_to_group[block] = group_id
            group_to_blocks.setdefault(group_id, []).append(block)
            group_id += 1

    # Přiřaď barvy skupinám
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    group_colors = {}
    for i, group in enumerate(group_to_blocks):
        group_colors[group] = color_list[i % len(color_list)]

    # --- Bloky vlevo ---
    fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(12, 6))
    x_min = min(block.x_min for block in blocks)
    x_max = max(block.x_min + block.width for block in blocks)
    y_min = min(block.y_min for block in blocks)
    y_max = max(block.y_min + block.height for block in blocks)
    for block in blocks:
        x = block.x_min
        y = block.y_min
        w = block.width
        h = block.height
        color = group_colors[block_to_group[block]]
        rect = plt.Rectangle((x, y), w, h, fill=True, edgecolor='black', facecolor=color, alpha=0.6)
        ax_blocks.add_patch(rect)
        ax_blocks.text(x + w / 2, y + h / 2, block.name, ha='center', va='center', fontsize=8)
    ax_blocks.set_xlim(x_min, x_max)
    ax_blocks.set_ylim(y_min, y_max)
    ax_blocks.set_aspect('equal')
    ax_blocks.set_title("Blocks positions")
    ax_blocks.set_xlabel('X')
    ax_blocks.set_ylabel('Y')

    # --- Vykreslení propojení (nets) ---
    if nets is not None:
        for net_name, net_blocks in nets:
            centers = [(b.x_min + b.width / 2, b.y_min + b.height / 2) for b in net_blocks]
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    x0, y0 = centers[i]
                    x1, y1 = centers[j]
                    ax_blocks.plot([x0, x1], [y0, y1], color='black', linewidth=2.5, alpha=0.9, zorder=1)

    # --- Strom vpravo (beze změny) ---
    def plot_node(node, x, y, depth, step=2.0, color='orange'):
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', zorder=2)
        ax_tree.add_patch(circle)
        ax_tree.text(x - 0.4, y - 0.4, node.name, ha='center', va='bottom', fontsize=10, zorder=3)
        if hasattr(node, 'x_child') and node.x_child:
            x_l = x + step
            y_l = y
            ax_tree.arrow(x, y, step, 0, head_width=0.15, head_length=0.5, fc='green', ec='green',
                          length_includes_head=True)
            plot_node(node.x_child, x_l, y_l, depth + 1, step)
        if hasattr(node, 'y_child') and node.y_child:
            x_r = x
            y_r = y + step
            ax_tree.arrow(x, y, 0, step, head_width=0.15, head_length=0.5, fc='blue', ec='blue',
                          length_includes_head=True)
            plot_node(node.y_child, x_r, y_r, depth + 1, step)

    plot_node(root, 0, 0, 1, color='darkgreen')
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()


################################################################################
if __name__ == "__main__":
    ############################################################################
    case = 1
    NETLIST_FILE = f"/Users/daliborbarri/dalbar/000_programming/python/ACDRC-ALDA_workshop_analog_placer_2025_04_4-6/testcases/case{case}.netlist"
    SYMMETRY_FILE = f"/Users/daliborbarri/dalbar/000_programming/python/ACDRC-ALDA_workshop_analog_placer_2025_04_4-6/testcases/case{case}.sym"
    BLOCK_FILE = f"/Users/daliborbarri/dalbar/000_programming/python/ACDRC-ALDA_workshop_analog_placer_2025_04_4-6/testcases/case{case}.block"
    OUTPUT_FILE = f"/Users/daliborbarri/dalbar/000_programming/python/ACDRC-ALDA_workshop_analog_placer_2025_04_4-6/testcases/case{case}.output"

    AR = 1  # (verical/horizontal resp. y/x)

    ############################################################################
    tree = dba_top_tree()
    tree.dba_load(NETLIST_FILE, SYMMETRY_FILE, BLOCK_FILE)
    # visualize_tree_and_blocks2(tree.root, tree.l_virtual_hier_w_hard_blocks)

    ############################################################################
    # Volání optimalizace, pokud existuje
    # Pokud není metoda optimize, pouze vypočítá cost
    import threading
    from utils import PA3Cost
    from utils import Timer  # nebo vlastní implementace Timeru
    import copy

    cost_func = PA3Cost(AR)
    timer = Timer(timeout=1)  # nastavte vhodný timeout

    mtx = threading.Lock()

    i = 1
    if hasattr(tree, "optimize"):
        tree.optimize(cost_func, timer)
    with mtx:
        print(f"The best one is -> Cost={cost_func(tree):.6f}")

    # tree.optimize(cost_func, "")

    ############################################################################
    # visualize_block_positions(tree.l_virtual_hier_w_hard_blocks)
    # visualize_blocksand_interconnections(tree.l_virtual_hier_w_hard_blocks, tree.nets)
    # visualize_tree(tree.root)
    # visualize_tree_and_blocks2(tree.root, tree.l_virtual_hier_w_hard_blocks, tree.l_nets)
    visualize_tree_and_blocks3(tree.root, tree.l_virtual_hier_w_hard_blocks)
    visualize_tree_and_blocks4(tree.root, tree.l_virtual_hier_w_hard_blocks, tree.l_nets)

    ############################################################################
    print(f"Optimized tree")
    tree.Dump(OUTPUT_FILE)

# dba TODO:
#  1) optimize function - maybe simulated annealing
#  2) packing - maybe better algorithm
#  3) perturbation - maybe better algorithm
#  4) def swap_without_link does not swap parent link (the box is empty)

