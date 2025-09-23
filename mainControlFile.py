from bStarTreeBuilder import BStarTreeBuilder
from initialPlacement import InitialPlacer
from placementVisualizer import PlacementVisualizer
from simulatedAnnealingOptimizer import SimulatedAnnealingOptimizer


def main():
    # wmi HINTME: change the name of the input file as needed
    input_filename = 'loadDevices_in01'
    bstar_output_filename = 'loadDevices_out01'
    placement_output_filename = 'initPlacement_out01'
    SA_output_filename = 'initSimulatedAnnealing_out01'
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


    #######################################################################################
    # Optimize Placement
    SAoptimizer = SimulatedAnnealingOptimizer()
    SAoptimizer.set_optimization_parameters(
        initial_temp=2000.0,
        final_temp=1.0,
        cooling_rate=0.98,
        max_iterations=2000,
        timeout_seconds=120,
        patience=500,
        aspect_ratio_target=2.0
    )
    SAoptimizer.main_local(placement_output_filename, SA_output_filename)
    #######################################################################################

    #######################################################################################
    # Visualization
    IPlacer.main_local(SA_output_filename, SA_output_filename)
    VisuAlize.main_local(SA_output_filename)
    #######################################################################################

if __name__ == "__main__":
    main()
