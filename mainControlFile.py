import sys
from filterInputJSON import JSONBlockFilter

def main():
    # wmi HINTME: change the name of the input file as needed
    input_filename = 'loadDevices_in01'

    # wmi TODO: change the code to be compatible with n8n JSON input and output
    # wmi TODO: do not filter data, just add new keys to the existing JSON structure
    # structure of the data flow has to be linear, no branches
    # each step takes JSON input and produces JSON output to be compatible with n8n

    #######################################################################################
    # Load unfiltered JSON file
    # Getting rid of the irrelevant data
    b1_filterInput = JSONBlockFilter()
    b1_filterInput.load_json_file(f"{input_filename}.json")
    b1_filterInput.extract_selected_keys()
    b1_filterInput.print_data()

    #######################################################################################

    #######################################################################################
    # Build B* tree

    # Returns B* tree structure in JSON format
    #######################################################################################

    #######################################################################################
    # Initial Placement

    # Returns JSON file with initial placement and B* tree structure
    #######################################################################################


if __name__ == "__main__":
    main()
