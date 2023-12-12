
import faiss
import json
import nltk
import numpy as np
import os

index = None
sentence_to_index_mapping = None
data_dir = 'D:/JupyterPrograms/0-CHAT_GPT/EXPERIMENTS/ML_CHAMP/data/'
index_name = data_dir + 'doc_index.index'
mapping_name = data_dir + 'sentence_to_index_mapping.json'

# This method will attempt to load an existing index and the sentence mappings.
# If they don't exist, it will rebuild them.
def load_index(df=None, model=None):
    global index
    global sentence_to_index_mapping
    
    if index is None:
        if os.path.exists(index_name) and os.path.exists(mapping_name):
            index = faiss.read_index(index_name)
            print(f"Loaded index from {index_name}")
            
            with open(mapping_name, 'r') as mapping_file:
                sentence_to_index_mapping = json.load(mapping_file)
            print(f"Loaded sentence_to_index_mapping from {mapping_name}")
        else:
            if not df or not model:
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

                print('sentence_embeddings.shape', sentence_embeddings.shape)   

                for sentence_idx, embedding in enumerate(sentence_embeddings):
                    # Add sentence embedding to the index
                    index.add(np.expand_dims(embedding, axis=0))

                    # Track the mapping between sentence index and its embedding index
                    sentence_to_index_mapping[str(len(sentence_to_index_mapping))] = {
                        'document_index': doc_id,
                        'sentence_index': sentence_idx,
                        'sentence_text': sentences[sentence_idx]  # Save the actual sentence
                    }

                print("Length of the index:", index.ntotal)
            
            # Save the index and mapping
            faiss.write_index(index, index_name)
            with open(mapping_name, 'w') as mapping_file:
                json.dump(sentence_to_index_mapping, mapping_file)

        print("Final length of the index:", index.ntotal)