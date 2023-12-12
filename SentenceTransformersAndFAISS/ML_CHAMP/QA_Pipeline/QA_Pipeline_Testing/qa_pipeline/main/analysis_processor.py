from collections import Counter
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
index_processor.load_index()

index = index_processor.index
sentence_to_index_mapping = index_processor.sentence_to_index_mapping

question_and_answer_history = []

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

def run_lmm_tests():
    # TASKS:
    # Generate question, query index, generate prompts for final LMM, get final answer
    # The Ensembles example has this in the code: if not pretrained_model_directory:
    # This can be substituted with questions and answers that have occurred in this 'conversation'
    print('Running LMM Tests...')
    questions = get_questions()
    for question in questions:
        # For each question, get the list of best sentence fragment matches from the index.
        most_similar_prompts = get_index_responses(question)
        
        ##########  PRINT STATEMENTS  ##########
#         print('USER QUERY:', question)
#         print("Most similar prompts to the user query:")
#         for doc_id, sentence_idx, sentence in most_similar_prompts:
#             result = data_loader.get_info_by_doc_id(doc_id) 
#             if result is not None:
#                 doc_idx = result['doc_id']
#                 filename = result['doc_filename']
#                 print(f"\nDocument Index: {doc_idx}\t Sentence Index: {sentence_idx}\t Filename: {filename}")
#                 print(f"Sentence:\n******************************\n\t{sentence}")
#                 print('******************************\n')
#             else:
#                 print("Document ID", doc_id, "associated with sentence_idx", sentence_idx, "not found in the DataFrame")
        ##########  PRINT STATEMENTS  ##########
    
        # Now that you have the most_similar_prompts, call the method to run the query
        answer = answer_query(question_and_answer_history, question, most_similar_prompts)

def answer_query(question_and_answer_history, question, most_similar_prompts):
    
    # Extract only the sentences from most_similar_prompts
    faiss_sentences = [sentence for _, _, sentence in most_similar_prompts]
    
    # Temporary until I find a better way to question all of the relevant documents
    # Extract doc_ids from most_similar_prompts
    doc_ids = [doc_id for doc_id, _, _ in most_similar_prompts]
    # Count occurrences of each doc_id
    doc_id_counts = Counter(doc_ids)
    # Find the doc_id with the highest count
    most_common_doc_id, occurrences = doc_id_counts.most_common(1)[0]
    doc_result = data_loader.get_info_by_doc_id(most_common_doc_id) 
    if doc_result is not None:
#         doc_id = doc_result['doc_id']
        doc_filename = doc_result['doc_filename']
        document = doc_result['doc_contents'] 
        print('most_common_doc_id:', most_common_doc_id, doc_filename)  
        answer = process_single_large_document(question_and_answer_history, question, faiss_sentences, document)
        print('\n\nUser question:', question, '\nAnswer:', answer)
        
def process_single_large_document(question_and_answer_history, user_question, faiss_sentences, large_document):
    qa_model, qa_tokenizer = model_settings.get_qa_model_and_tokenizer()
    
    # Concatenate FAISS sentences with the user question to form the context
    faiss_context = ' '.join(faiss_sentences)
    context = user_question + " " + faiss_context

    # Tokenize the context and large document separately
    tokenized_context = qa_tokenizer.encode_plus(context, return_tensors="pt", max_length=512, truncation=True)
    tokenized_document = qa_tokenizer.encode_plus(large_document, return_tensors="pt", max_length=512, truncation=True)

    # Trim or truncate the tokenized document to fit within the remaining space
    max_seq_len = qa_tokenizer.model_max_length - tokenized_context['input_ids'].size(1) - 3  # Account for special tokens [CLS], [SEP], etc.
    input_ids_doc = tokenized_document['input_ids'][:, :max_seq_len]
    attention_mask_doc = tokenized_document['attention_mask'][:, :max_seq_len]

    # Concatenate the tokenized context and truncated document
    input_ids = torch.cat([tokenized_context['input_ids'], input_ids_doc], dim=1)
    attention_mask = torch.cat([tokenized_context['attention_mask'], attention_mask_doc], dim=1)

    # Answer the question based on the combined context and document
    with torch.no_grad():
        outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Process outputs to get the answer
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    # Decode the tokens into the answer text
    answer = qa_tokenizer.decode(input_ids[0][start_idx:end_idx+1], skip_special_tokens=True)
    
    return answer
    

def get_questions():
    questions = ["How is global health affected by climate change?"]
    return questions

def get_index_responses(user_query):
    index_transformer_model, _  = model_settings.get_index_transformer_model_and_tokenizer()
    # Encode user query into an embedding
    user_query_embedding = index_transformer_model.encode(user_query, convert_to_tensor=True).numpy()
    
    # Search in FAISS index
    k = 5  # Number of most similar prompts to retrieve
    D, I = index.search(np.array([user_query_embedding]), k)

    # Retrieve sentences and corresponding documents based on valid indices
    most_similar_prompts = []
    for idx_num in I[0]:
        idx = str(idx_num)
        if idx in sentence_to_index_mapping:
            mapping = sentence_to_index_mapping[idx]
            doc_idx = mapping['document_index']
            sentence_idx = mapping['sentence_index']
            sentence = mapping['sentence_text']
            most_similar_prompts.append((doc_idx, sentence_idx, sentence))
            
    return most_similar_prompts
    

def run_various_components():
    df = data_loader.get_document_index()
     
#     # Generate a list of 'questions' from the document directives
#     topic_names = dp.get_topic_names()
#     print('TOPIC NAMES:', topic_names)
#     topic_titles = dp.get_topic_titles()
#     print('\nTOPIC TITLES:\n', topic_titles)
#     print('\nAI DIRECTIVES:\n', dp.get_topic_text('AI'))
    
    index_transformer_model, _  = model_settings.get_index_transformer_model_and_tokenizer()
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
        
        
    