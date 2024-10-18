import json

# Define the path to your input text file and output JSON file
input_file_path = 'qa.txt'  # Change this to your input file path
output_file_path = 'output.json'  # Change this to your desired output file path

# Initialize a list to hold the question-answer pairs
data = []

# Read the input text file
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        # Split the line into question and answer parts
        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()  # Get the question
            # Create a dictionary for the question
            entry = {
                "question": question,
                "answer_1": "",
                "answer_2": "",
                "answer_3": ""
            }
            data.append(entry)

# Write the data to a JSON file
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Data successfully written to {output_file_path}")
