
import faiss
import json
import nltk
import numpy as np
import os
import pandas as pd
import uuid

from sentence_transformers import SentenceTransformer

sentences_index = None
sentence_to_index_mapping = None
data_dir = 'D:/JupyterPrograms/0-CHAT_GPT/EXPERIMENTS/ML_CHAMP/data/'

index_name = os.path.join(data_dir, 'sentences.index')
mapping_name = os.path.join(data_dir, 'sentence_to_index_mapping.json')

index_transformer_model = None
index_transformer_model_name = 'sentence-transformers/all-mpnet-base-v2'
index_transformer_tokenizer = None

docs_dir = os.path.join(data_dir, 'docs')
document_guid_lookup_csv = os.path.join(data_dir, 'document_guid_lookup.csv')
document_guid_lookup_df = None

def get_document_guid_lookup():
    global document_guid_lookup_df
    # See if the DataFrame already exists.  Load it or create it if it doesn't    
    if document_guid_lookup_df is None:
        # Check if the CSV file exists
        if not os.path.exists(document_guid_lookup_csv):
            # If the file doesn't exist, create a new DataFrame
            # Create an empty DataFrame with column names
            column_names = ['doc_guid', 'doc_filename']
            document_guid_lookup_df = pd.DataFrame(columns=column_names)

            # Save the DataFrame to the CSV file
            document_guid_lookup_df.to_csv(document_guid_lookup_csv, index=False)
            print(f"CSV file of size {len(document_guid_lookup_df)} created at: {document_guid_lookup_csv}")
        else:
            # If the file already exists, load it into a DataFrame
            document_guid_lookup_df = pd.read_csv(document_guid_lookup_csv)
            print(f"CSV file already exists at: {document_guid_lookup_csv}")
            print("Loaded existing DataFrame of size", len(document_guid_lookup_df))
            
    return document_guid_lookup_df

def get_document_info(doc_id):
    global document_guid_lookup_df
    filename = None
    document = None
    guid_df = document_guid_lookup_df.loc[document_guid_lookup_df['doc_guid'] == doc_id, 'doc_filename']
    if len(guid_df) > 0:
        filename = guid_df.values[0]
    
    file_path = os.path.join(docs_dir, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()            
    return filename, document    

def add_document_to_index(guid, file_name):   
    global sentences_index
    global sentence_to_index_mapping
    global index_transformer_model
#     print("\n\nStarting add_document_to_index().  Length of the index:", sentences_index.ntotal)
    
    file_path = os.path.join(docs_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()

        sentences = nltk.sent_tokenize(document)
        sentence_embeddings = index_transformer_model.encode(sentences, convert_to_tensor=True)                  

        for sentence_idx, embedding in enumerate(sentence_embeddings):
            # Add sentence embedding to the index
            sentences_index.add(np.expand_dims(embedding, axis=0))

            # Track the mapping between sentence index and its embedding index
            sentence_to_index_mapping[len(sentence_to_index_mapping)] = {
                'document_index': guid,
                'sentence_index': sentence_idx,
                'sentence_text': sentences[sentence_idx]  # Save the actual sentence
            }

#         print("Length of the index:", sentences_index.ntotal)
    return sentences_index, sentence_to_index_mapping

def update_index_and_mappings(new_entries):   
    global sentences_index
    global sentence_to_index_mapping
    
    for record in new_entries:
        guid = record['doc_guid']
        file_name = record['doc_filename']
        print('Updating', guid, file_name)
        sentences_index, sentence_to_index_mapping = add_document_to_index(guid, file_name)
            
    # Save the index and mapping
#     print("PREPARING TO SAVE THE INDEX OF LENGTH:", sentences_index.ntotal)
    faiss.write_index(sentences_index, index_name)
    with open(mapping_name, 'w') as mapping_file:
        json.dump(sentence_to_index_mapping, mapping_file)

# TODO: This should cause an update of the FAISS index and sentence_to_index_mappings file
def update_document_guid_lookup_df():
    global document_guid_lookup_df
#     print('Called update_document_guid_lookup_df...')
    if document_guid_lookup_df is None:
        # Load the existing one or create a new one
        get_document_guid_lookup()
    
    # Get the list of files in the documents directory
    existing_files = set(document_guid_lookup_df['doc_filename'].tolist())
    files_to_process = [file for file in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, file))]
#     print('existing_files:', existing_files)
#     print('os.listdir(docs_dir):', os.listdir(docs_dir))
    
    # Check for files that are not present in the DataFrame and add them
    new_entries = []
    for file_name in files_to_process:
        if file_name not in existing_files:
            # Generate a unique ID for the new document
            unique_id = str(uuid.uuid4())

            # Prepare a new entry for the DataFrame
            new_entry = {'doc_guid': unique_id, 'doc_filename': file_name}
            new_entries.append(new_entry)

    # Concatenate new entries to the existing DataFrame
    if new_entries:
        print('There are new entries:', new_entries)
        new_df = pd.DataFrame(new_entries)
        document_guid_lookup_df = pd.concat([document_guid_lookup_df, new_df], ignore_index=True)

        # Save the updated DataFrame to the CSV file
        update_index_and_mappings(new_entries)
        document_guid_lookup_df.to_csv(document_guid_lookup_csv, index=False)
        print("Documents checked and DataFrame updated successfully.")
    else:
        print('No new documents were found.  No DataFrame update required.')

# For now, just provide some hard-coded model retrieval methods
def get_index_transformer_model_and_tokenizer():
    global index_transformer_model
    global index_transformer_tokenizer
    if index_transformer_model is None:
        # Load it
        index_transformer_model = SentenceTransformer(index_transformer_model_name)
    return index_transformer_model, index_transformer_tokenizer

# Allow the model and tokenizer to be stored on Jupyter and pushed into memory instead of reloading it.
def set_index_transformer_model_and_tokenizer(model, tokenizer):
    global index_transformer_model
    global index_transformer_model_name
    global index_transformer_tokenizer
#     print('SETTING the index transformer model and tokenizer...')
    index_transformer_model = model
#     index_transformer_model_name = model_name
    index_transformer_tokenizer = tokenizer

def load_existing_index():
    global document_guid_lookup_df
    global sentences_index
    global sentence_to_index_mapping
    if os.path.exists(index_name) and os.path.exists(mapping_name):
        sentences_index = faiss.read_index(index_name)
#         print(f"Loaded EXISTING index from {index_name}.  Size is", sentences_index.ntotal)

        with open(mapping_name, 'r') as mapping_file:
            sentence_to_index_mapping = json.load(mapping_file)
        print(f"Loaded sentence_to_index_mapping from {mapping_name}")
    else:
        print('Index does not exist.  Attempting to build it.')
        # Populate the index and track the sentence ids and locations
        sentences_index = faiss.IndexFlatL2(768)  # Create an index
        # Maintain a mapping between sentence embeddings' index and their original sentences
        sentence_to_index_mapping = {}
        
        # Load the docs into the index
        for _, row in document_guid_lookup_df.iterrows():
            guid = row['doc_guid']
            file_name = row['doc_filename']
            add_document_to_index(guid, file_name)
            
        # Save the index and mapping
        faiss.write_index(sentences_index, index_name)
        with open(mapping_name, 'w') as mapping_file:
            json.dump(sentence_to_index_mapping, mapping_file)
    
#     print("Length of the index:", sentences_index.ntotal)
    return sentences_index, sentence_to_index_mapping

# This method will attempt to load an existing index and the sentence mappings.
# If they don't exist, it will rebuild them.
def load_index(df=None, model=None):
    global sentences_index
    global sentence_to_index_mapping
    
    if index is None:
        if os.path.exists(index_name) and os.path.exists(mapping_name):
            index = faiss.read_index(index_name)
            print(f"Loaded index from {index_name}")
            
            with open(mapping_name, 'r') as mapping_file:
                sentence_to_index_mapping = json.load(mapping_file)
            print(f"Loaded sentence_to_index_mapping from {mapping_name}")
        else:
            if df is None or not model:
                raise Exception('An existing index was not found and the dataframe and/or index model were not supplied.')
                
            # Populate the index and track the sentence ids and locations
            index = faiss.IndexFlatL2(768)  # Create an index
            # Maintain a mapping between sentence embeddings' index and their original sentences
            sentence_to_index_mapping = {}

            # Load the docs into the index
            for idx, row in df.iterrows():
                doc_id = row['doc_id']
                doc_filename = row['doc_filename']
                doc = row['doc_contents']
                sentences = nltk.sent_tokenize(doc)
                sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

#                 print('sentence_embeddings.shape', sentence_embeddings.shape)   

                for sentence_idx, embedding in enumerate(sentence_embeddings):
                    # Add sentence embedding to the index
                    index.add(np.expand_dims(embedding, axis=0))

                    # Track the mapping between sentence index and its embedding index
                    sentence_to_index_mapping[str(len(sentence_to_index_mapping))] = {
                        'document_index': doc_id,
                        'sentence_index': sentence_idx,
                        'sentence_text': sentences[sentence_idx]  # Save the actual sentence
                    }

#                 print("Length of the index:", index.ntotal)
            
            # Save the index and mapping
            faiss.write_index(index, index_name)
            with open(mapping_name, 'w') as mapping_file:
                json.dump(sentence_to_index_mapping, mapping_file)

        print("Final length of the index:", index.ntotal)

def get_index_responses(user_query, similarity_threshold=0.25):
    global sentences_index
    global index_transformer_model
    global sentence_to_index_mapping
    
    # Encode user query into an embedding
    user_query_embedding = index_transformer_model.encode(user_query, convert_to_tensor=True).numpy()
    
    # Search in FAISS index
    k = 50  # Number of most similar prompts to retrieve
    D, I = sentences_index.search(np.array([user_query_embedding]), k)
    
    
    # Filter responses based on similarity threshold
    most_similar_prompts = []
    filtered_responses = []
    count = 0
#     print('\nRetrieving index responses:')
    for distance, idx_num in zip(D[0], I[0]):
        similarity_score = 1 - distance  # Calculate similarity score
        if similarity_score >= similarity_threshold:
            idx = str(idx_num)
            if idx in sentence_to_index_mapping:
                count += 1
                mapping = sentence_to_index_mapping[idx]
                doc_idx = mapping['document_index']
                sentence_idx = mapping['sentence_index']
                sentence = mapping['sentence_text']
#                 if count < 10:
#                     print('\t', similarity_score, ': ', sentence[:60])
                most_similar_prompts.append((doc_idx, sentence_idx, sentence))
            
    print('\nThere were', len(most_similar_prompts), 'sentences found out of a possible', k, 'that were above the threshold of', similarity_threshold)
    if len(most_similar_prompts) == 0:
        # Keep drilling down if you don't find anything.
        new_similarity_threshold = similarity_threshold*0.5
        print('No prompts found.  Updating similarity_threshold to', new_similarity_threshold)
        return get_index_responses(user_query, new_similarity_threshold)
    
    return most_similar_prompts
        
# Default startup methods:
# Auto-load the document_guid_lookup_df
get_document_guid_lookup()

get_index_transformer_model_and_tokenizer()

# Autoload the sentences_index and sentence_to_index_mapping
load_existing_index()
