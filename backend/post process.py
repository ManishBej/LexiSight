import json

def process_dataset():
    # Path to the JSON file
    file_path = './MANISH.json'
    
    # Read the JSON file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Initialize case counter
    case_number = 0
    
    # Process each entry in the dataset
    for i in range(len(data)):
        # Only process non-empty dictionary entries with a case_id
        if isinstance(data[i], dict) and data[i] and 'case_id' in data[i]:
            case_number += 1
            
            # Remove pdf_file field if it exists
            if 'pdf_file' in data[i]:
                del data[i]['pdf_file']
            
            # Update case_id with sequential numbering
            data[i]['case_id'] = f"CASE_{case_number:05d}"
    
    # Write the modified data back to the file with UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)
    
    print(f"Successfully processed {case_number} cases.")
    print(f"Removed 'pdf_file' parameter and renumbered 'case_id' from CASE_00001 to CASE_{case_number:05d}")

# Execute the processing function
if __name__ == "__main__":
    process_dataset()