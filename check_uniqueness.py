import pandas as pd
from collections import Counter

def find_and_remove_repeated_sentences(file_path, sheet_name):
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Combine all text into one list with their positions (row, column)
    sentences_with_positions = []
    for column in df.columns:
        sentences_with_positions.extend(((i, column), str(value)) for i, value in df[column].dropna().items())

    # Count the occurrences of each sentence
    sentence_counts = Counter(sentence for _, sentence in sentences_with_positions)
    
    # Find sentences that are repeated more than once
    repeated_sentences = {sentence: count for sentence, count in sentence_counts.items() if count > 1}

    if repeated_sentences:
        print("Repeated sentences:")
        for sentence, count in repeated_sentences.items():
            print(f'"{sentence}" is repeated {count} times')
        
        # Create a set to keep track of the first occurrences
        first_occurrences = set()

        # Remove rows containing repeated sentences (except the first occurrence)
        for (index, column), sentence in sentences_with_positions:
            if sentence in repeated_sentences:
                if sentence not in first_occurrences:
                    first_occurrences.add(sentence)
                else:
                    df.at[index, column] = None

        # Save the modified DataFrame back to the Excel file
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
        print("Repeated sentences removed and Excel file updated.")
    else:
        print("No repeated sentences found.")

# Example usage
file_path = '/home/urte/lithuanian_task_2_collection.xlsx'  # Replace with your Excel file path
sheet_name = 'Sheet1'  # Replace with your sheet name
find_and_remove_repeated_sentences(file_path, sheet_name)
