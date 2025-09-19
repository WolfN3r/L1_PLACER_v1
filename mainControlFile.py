from bStarTreeBuilder import BStarTreeBuilder
from initialPlacement import InitialPlacer
from placementVisualizer import PlacementVisualizer


def main():
    # wmi HINTME: change the name of the input file as needed
    input_filename = 'loadDevices_in01'
    bstar_output_filename = 'loadDevices_out01'
    placement_output_filename = 'initPlacement_out01'
    # wmi TODO: change the code to be compatible with n8n JSON input and output
    # wmi TODO: do not filter data, just add new keys to the existing JSON structure
    # structure of the data flow has to be linear, no branches
    # each step takes JSON input and produces JSON output to be compatible with n8n

    #######################################################################################
    # Build B* tree
    BTree = BStarTreeBuilder()
    BTree.main_local(input_filename, bstar_output_filename)
    # Returns B* tree structure in JSON format
    #######################################################################################

    #######################################################################################
    # Initial Placement
    IPlacer = InitialPlacer()
    IPlacer.main_local(bstar_output_filename, placement_output_filename)
    # Returns JSON file with initial placement and B* tree structure
    #######################################################################################

    #######################################################################################
    # Visualization
    VisuAlize = PlacementVisualizer()
    VisuAlize.main_local(placement_output_filename)
    #######################################################################################

if __name__ == "__main__":
    main()
