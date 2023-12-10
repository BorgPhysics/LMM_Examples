import numpy as np
import os
import pandas as pd
import pprint
import torch
from torch import Tensor

from ..data import data_loader
from ..directives import directives_processor
from ..models import model_settings
from ..faiss_index import faiss_index_processes as index_processor

dp = directives_processor.DirectivesProcessor()

def run_preprocessing_commands(preprocessing_commands):
    
    # Loop through the pretraining commands to get the things that we need
    for preprocessing_command in preprocessing_commands:
        if preprocessing_command.get('check_for_new_docs', False):
            data_loader.update_document_index()
        
#         ts = preprocessing_command.get('test_size', None)
#         rs = preprocessing_command.get('random_state', None)
#         if ts:
#             test_size = ts
#         elif rs:
#             random_state = rs
#         if not data:
#             data, target, target_column_name, descr = datasource_processor.get_data(preprocessing_command)

def run_various_components():
    df = data_loader.get_document_index()      
     
#     # Generate a list of 'questions' from the document directives
#     topic_names = dp.get_topic_names()
#     print('TOPIC NAMES:', topic_names)
#     topic_titles = dp.get_topic_titles()
#     print('\nTOPIC TITLES:\n', topic_titles)
#     print('\nAI DIRECTIVES:\n', dp.get_topic_text('AI'))
    
    index_transformer_model = model_settings.get_index_transformer_model()
#     print('index_transformer_model:\n\t', index_transformer_model)
#     summarization_model, summarization_tokenizer = model_settings.get_summarization_model_and_tokenizer()
#     print('summarization_model:\n\t', summarization_model)
#     qa_model, qa_tokenizer = model_settings.get_qa_model_and_tokenizer()
#     print('qa_model:\n\t', qa_model) 
    
    
    index_processor.load_index(df, index_transformer_model)
    index = index_processor.index
    sentence_to_index_mapping = index_processor.sentence_to_index_mapping
    
#     print('sentence_to_index_mapping:\n\t')
#     pp = pprint.PrettyPrinter(indent=4)
# #     pp.pprint(sentence_to_index_mapping)
    
#     # OR - Check the first 10 entries
#     # Slice the first 5 items of the dictionary
#     first_5_items = {k: sentence_to_index_mapping[k] for k in sorted(sentence_to_index_mapping)[:5]}
#     pp.pprint(first_5_items)
    
#     mapping = sentence_to_index_mapping.get("790")
#     print('mapping:', mapping)
    
    test_user_query(df, index_transformer_model, index, sentence_to_index_mapping)
    
    
#     print('The document index contains', len(df), 'records.\n\t', df.head(1))
    
#     doc_id = 'e5e9975d-5826-4664-83a7-92115931b302'
#     result = data_loader.get_info_by_doc_id(doc_id)    
#     if result is not None:
#         doc_id = result['doc_id']
#         doc_filename = result['doc_filename']
#         doc_contents = result['doc_contents']
# #         print(f"Document ID: {doc_id}")
# #         print(f"Document Filename: {doc_filename}")
# #         print(f"Document Contents: {doc_contents[:150]}")
#         summary = model_settings.summarize_text(doc_contents)
#         print(doc_id, 'summary:\n\t', summary)
#     else:
#         print("Document ID", doc_id, "not found in the DataFrame")
    
#     # Summarize each document using DataFrame
#     for index, row in df.iterrows():
#         doc_id = row['doc_id']
#         doc_filename = row['doc_filename']
#         doc_contents = row['doc_contents']

#         print(f"Document ID: {doc_id}, File Name: {doc_filename}")
#         print('******************************')

#         # Get the content from the DataFrame and summarize it
#         summary = model_settings.summarize_text(doc_contents)
#         print('Summary:\n\t', summary)
#         print('******************************\n')
    
def test_user_query(df, model, index, sentence_to_index_mapping):
    # Run a user query against the index and get the top 5
    # User query or question
    user_query = "How does AI affect global health?"
    user_query = "How is global health affected by climate change?"

    # Encode user query into an embedding
    user_query_embedding = model.encode(user_query, convert_to_tensor=True).numpy()

    # Search in FAISS index
    k = 5  # Number of most similar prompts to retrieve
    D, I = index.search(np.array([user_query_embedding]), k)
    print(D, I)

    # Retrieve sentences and corresponding documents based on valid indices
    most_similar_prompts = []
    for idx in I[0]:
        print('Searching for', idx, 'in sentence_to_index_mapping.')
        mapping = sentence_to_index_mapping.get(str(idx))
        doc_idx = mapping['document_index']
        sentence_idx = mapping['sentence_index']
        sentence = mapping['sentence_text']
        most_similar_prompts.append((doc_idx, sentence_idx, sentence))

    print('\nUSER QUERY:', user_query)
    print("Most similar prompts to the user query:")
#     print('\t', most_similar_prompts)
    for doc_id, sentence_idx, sentence in most_similar_prompts:
        result = data_loader.get_info_by_doc_id(doc_id) 
        if result is not None:
            doc_id = result['doc_id']
            filename = result['doc_filename']
            print(f"\nDocument Index: {doc_idx}\t Sentence Index: {sentence_idx}\t Filename: {filename}")
            print(f"Sentence:\n******************************\n\t{sentence}")
            print('******************************\n')
        else:
            print("Document ID", doc_id, "associated with sentence_idx", sentence_idx, "not found in the DataFrame")
        
        
    