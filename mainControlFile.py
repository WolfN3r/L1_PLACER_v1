from bStarTreeBuilder import BStarTreeBuilder


def main():
    # wmi HINTME: change the name of the input file as needed
    input_filename = 'loadDevices_in01'
    output_filename= 'loadDevices_out01'
    # wmi TODO: change the code to be compatible with n8n JSON input and output
    # wmi TODO: do not filter data, just add new keys to the existing JSON structure
    # structure of the data flow has to be linear, no branches
    # each step takes JSON input and produces JSON output to be compatible with n8n

    #######################################################################################
    # Build B* tree
    BTree = BStarTreeBuilder()
    BTree.main_local('loadDevices_in01', 'loadDevices_out01')
    # Returns B* tree structure in JSON format
    #######################################################################################

    #######################################################################################
    # Initial Placement

    # Returns JSON file with initial placement and B* tree structure
    #######################################################################################


if __name__ == "__main__":
    main()
