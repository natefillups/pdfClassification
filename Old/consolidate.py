def consolidate_page_numbers(file_paths):
    page_numbers_dict = {}
    
    # Read each file and consolidate page numbers
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) == 3:
                    file_name = parts[0]
                    page_number = parts[2]
                    # If the file name is not in the dictionary, add it
                    if file_name not in page_numbers_dict:
                        page_numbers_dict[file_name] = set()
                    # Add page number to the set
                    page_numbers_dict[file_name].add(page_number)
    
    # Construct lines with consolidated page numbers
    lines = []
    for file_name, page_numbers in page_numbers_dict.items():
        lines.append(f"{file_name} {' '.join(sorted(page_numbers))}")
    
    return lines

# Run both scripts and get outputs
file_paths = ["logs/model1.txt", "logs/model3.txt"]
consolidated_lines = consolidate_page_numbers(file_paths)

# Output the consolidated lines
for line in consolidated_lines:
    print(line)

