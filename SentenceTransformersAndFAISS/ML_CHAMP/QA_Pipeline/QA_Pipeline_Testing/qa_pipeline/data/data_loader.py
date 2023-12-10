import os
import pandas as pd
import uuid

data_dir = 'D:/JupyterPrograms/0-CHAT_GPT/EXPERIMENTS/ML_CHAMP/data/'
docs_dir = data_dir + 'docs'
docs_csv = data_dir + 'document_index.csv'
document_index_df = None

def get_document_index():
    global document_index_df
    # See if the DataFrame already exists.  Load it or create it if it doesn't
    
    if document_index_df is None:
        # Check if the CSV file exists
        if not os.path.exists(docs_csv):
            # If the file doesn't exist, create a new DataFrame
            # Create an empty DataFrame with column names
            column_names = ['doc_id', 'doc_filename', 'doc_contents']
            document_index_df = pd.DataFrame(columns=column_names)

            # Save the DataFrame to the CSV file
            document_index_df.to_csv(docs_csv, index=False)
            print(f"CSV file created at: {docs_csv}")
        else:
            # If the file already exists, load it into a DataFrame
            document_index_df = pd.read_csv(docs_csv)
            print(f"CSV file already exists at: {docs_csv}")
            print("Loaded existing DataFrame:")
            
    return document_index_df

# Function to retrieve doc_filename and doc_contents based on doc_id
def get_info_by_doc_id(doc_id):
    global document_index_df
    if document_index_df is None:
        get_document_index()
        
    # Check if doc_id exists in the DataFrame
    if doc_id in document_index_df['doc_id'].values:
        # Using .loc[] to get the information based on doc_id
        info = document_index_df.loc[document_index_df['doc_id'] == doc_id, ['doc_id', 'doc_filename', 'doc_contents']]
        return info.iloc[0]  # Return the information as a Series
    else:
        return None  # Return None if doc_id is not found

# Function to read file contents
def read_file_contents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    return contents

def update_document_index():    
    df = get_document_index()
    # Get the list of files in the documents directory
    existing_files = set(df['doc_filename'].tolist())
    files_to_process = [file for file in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, file))]
    
    # Check for files that are not present in the DataFrame and add them
    new_entries = []
    for file_name in files_to_process:
        if file_name not in existing_files:
            # Generate a unique ID for the new document
            unique_id = str(uuid.uuid4())

            # Read file contents
            file_path = os.path.join(docs_dir, file_name)
            contents = read_file_contents(file_path)

            # Prepare a new entry for the DataFrame
            new_entry = {'doc_id': unique_id, 'doc_filename': file_name, 'doc_contents': contents}
            new_entries.append(new_entry)

    # Concatenate new entries to the existing DataFrame
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        df = pd.concat([df, new_df], ignore_index=True)

        # Save the updated DataFrame to the CSV file
        df.to_csv(docs_csv, index=False)
        print("Documents checked and DataFrame updated successfully.")
    else:
        print('No new documents were found.  No DataFrame update required.')
    
