import json
import sys

class JSONBlockFilter:
    def __init__(self):
        self.original_data = None
        self.filtered_data = None

    def load_json_file(self, input_filename):
        """Loads a JSON file and stores the data in self.original_data."""
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                self.original_data = json.load(f)
            self.filtered_data = None  # reset when loading new file
            return self.original_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            self.original_data = None
            self.filtered_data = None
            return None

    def extract_selected_keys(self):
        """Stores only selected keys and their values in self.filtered_data."""
        if not self.original_data:
            print("No data loaded.")
            return None

        selected_keys = ['blocks']# , 'netlist', 'nets', 'layout_constraints']
        self.filtered_data = {key: self.original_data[key] for key in selected_keys if key in self.original_data}
        return self.filtered_data

    def print_data(self):
        # print filtered data
        if self.filtered_data is not None:
            print(json.dumps(self.filtered_data, indent=2, ensure_ascii=False))
        else:
            print("No data loaded.")

    def get_filtered_data(self):
        # return filtered data
        return self.filtered_data

    def get_original_data(self):
        # return original data
        return self.original_data

    #######################################################################################
    # n8n integration methods
    #######################################################################################
    def load_from_n8n(self):
        """Loads JSON data from stdin (for n8n integration)."""
        try:
            self.original_data = json.load(sys.stdin)
            self.filtered_data = None
            return self.original_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from stdin: {e}", file=sys.stderr)
            self.original_data = None
            self.filtered_data = None
            return None


    def output_to_n8n(self, filtered=True):
        """Outputs data to stdout (for n8n integration)."""
        data = self.filtered_data if filtered else self.original_data
        if data is not None:
            print(json.dumps(data))
        else:
            print("No data loaded.", file=sys.stderr)


    def main(self):
        """Main method for n8n integration."""
        self.load_from_n8n()
        self.extract_selected_keys()
        self.output_to_n8n(filtered=True)

    #######################################################################################


#######################################################################################
# main function for n8n integration
#######################################################################################

if __name__ == "__main__":
    filter_obj = JSONBlockFilter()
    filter_obj.main()