from collections import Counter
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pprint
import random
from torch import Tensor
from sentence_transformers import util

from ..directives import directives_processor
from ..models import model_settings
from ..faiss_index import faiss_index_processes as index_processor

dp = directives_processor.DirectivesProcessor()

# Misc variables
history_record = None
similarity_threshold = 0.5
qa_num_return_sequences=1
initial_question = None
question_list = []

# Formerly question_and_answer_history
# This has to track the history of the user question AND which methodology is being used
conversation_histories = {}

# Output tracking:
global_score_df = pd.DataFrame(columns=['dir', 'Question', 'Methodology', 'score1', 'score2', 'score3', 'average_score'])
# cos_sim is an array with a length equal to the number of generated_summaries (for each question).
#     -  This will vary depending on unique doc count
# best_summary_to_final_score will be a single value for each question
# question_to_final_answer_score will also be a single value for each question

# The previous cos_sim and question_to_final_answer_score values will be averages
# the previous best_summary_to_final_score will be a single value
# Need to also add a worst_summary_to_final_score

# methodologies = ['Question Only', 'Question & FAISS Vector', 'Question & FAISS Vector Summaries', 
#                  'Question & Generated Summaries', 'Question & Related Docs', 'Question, FAISS Vector & Related Docs', 
#                  'Question, FAISS Vector Summaries & Related Docs', 'Question, Generated Summaries & Related Docs']
# methodologies = ['Question Only', 'Question & FAISS Vector']
methodologies = ['Question & FAISS Vector']
# methodologies = ['Question Only', 'Question & FAISS Vector', 'Question & FAISS Vector Summaries']
# methodologies = ['Question & Related Docs']
# methodologies = ['Question, FAISS Vector & Related Docs']
# methodologies = ['Question, FAISS Vector Summaries & Related Docs']
# methodologies = ['Question Only', 'Question & FAISS Vector', 'Question & FAISS Vector Summaries', 
#                  'Question & Related Docs', 'Question, FAISS Vector & Related Docs', 
#                  'Question, FAISS Vector Summaries & Related Docs']

methodology_descriptions = {"Question Only":"This methodology submits the question directly to the final LMM without any additional context",  
                            "Question & Related Docs":"This methodology submits the question to the FAISS index to determine its related docs.\nIt then uses the question and related docs as context to the final LMM.", 
                            "Question & FAISS Vector":"This methodology submits the user question to the FAISS index to find all of the related sentences within the assigned parameter ranges.  It then uses the question and FAISS sentences as context for the final LMM.", 
                            "Question, FAISS Vector & Related Docs":"This methodology is the same as the Question & FAISS Vector methodology but also includes the FAISS-related docs in the context for the final LMM.", 
                            "Question & FAISS Vector Summaries":"This methodology generates a summary from all of the returned FAISS sentences.\nThe user question and summary are used as context for the final LMM.", 
                            "Question, FAISS Vector Summaries & Related Docs":"This methodology ", 
                            "Question & Generated Summaries":"This methodology ",
                            "Question, Generated Summaries & Related Docs":"This methodology "
}

def save_historical_output(*args, sub_directory=None, filename=None, display_on_screen=True):
    global history_record
    output = ' '.join(map(str, args))
    if history_record:
        # Save the output to the archived file
        if not filename:
            filename = "general_output.txt"
            
        output_directory = history_record.artifact_directory        
        if sub_directory:
            output_directory = os.path.join(output_directory, sub_directory)
            # artifacts directory will exist but this one might not
            os.makedirs(output_directory, exist_ok=True)        
        
        file_path = os.path.join(output_directory, filename)
        
        # Append text to the file
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(output + '\n')        
    
    if display_on_screen:
        print(str(output)[:300])

def run_postprocessing(postprocessing_commands):
    global history_record
    for postprocessing_command in postprocessing_commands:
        print_statement = postprocessing_command.get('print', None)
        if print_statement:
            print('The user asked the following question:', print_statement)
        next_question = postprocessing_command.get('next_question', None)
        if next_question:
            run_lmm_tests(next_question=next_question)

def run_preprocessing_commands(preprocessing_commands):
    global question_list
    global initial_question
    global sentences_index
    global sentence_to_index_mapping
    
    global similarity_threshold
    global qa_num_return_sequences
    
    # Loop through the pretraining commands to get the things that we need
    for preprocessing_command in preprocessing_commands:
        if preprocessing_command.get('check_for_new_docs', False):
            index_processor.update_document_guid_lookup_df()
            
        sim_thresh = preprocessing_command.get('similarity_threshold', None)
        if sim_thresh:
            print('setting sim_thresh to', sim_thresh)
            similarity_threshold = sim_thresh
            
        qa_gen_count = preprocessing_command.get('number_of_qa_sentences_to_generate', None)
        if qa_gen_count:
            qa_num_return_sequences = qa_gen_count
            
        question = preprocessing_command.get('initial_question', None)
        if question:
            initial_question = question
            if question not in question_list:
                question_list.append(question)
            
# START POINT
# This is the current entry point for 'training'
def run_processor_tests(hist_record=None):
    global history_record
    global question_list
    global global_score_df
    
    if hist_record:
        history_record = hist_record
    
    # TODO: You need a clean process to run multiple sentences and conversations using different methodologies
    # For now, just work on the organization of the print statements and output files.
    if len(question_list) == 0:
        question_list = get_questions()
        save_historical_output('No initial questions found, loaded stored questions.')
    save_historical_output('Processing questions.\n\t', question_list)
        
        
    for question_id, user_question in enumerate(question_list):
        # Keep in mind that you need to track statistics for each question and for the entire run        
        process_question(question_id, user_question)
        generate_question_stats(question_id)        
    
    generate_overall_stats()
    
    print('\n\n')
    
def process_question(question_id, user_question):
    # In this section, we process everything related to a specific question
    # Eight options here:
    #    - SUMMARIES:
    #        - No summaries                                                    
    #              - Q only
    #              - Q & V (just the returned vectors - no vector summary)
    #        - Summaries come from vectors                                     VS
    #        - Summaries come from LMM-generated text of vectors               GS
    #    - For each SUMMARY TYPE:
    #        - Submit question and summary to final LMM                        Q only, Q & V, Q & VS, Q & GS
    #        - Submit question, summary and relevant documents to final LMM    Q & Docs, Q, V & Docs, Q, VS, & Docs, Q, GS & Docs
    
    # Start simple - put the question in its proper directory
    save_historical_output('NOTE: All methodology types are used as an input context to the final LMM model.\n', 
                           sub_directory=str(question_id), filename='question_stats.log')
    
    most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)        
        
    doc_ids = [doc_id for doc_id, _, _ in most_similar_faiss_sentences]
    unique_doc_ids = set(doc_ids)
    save_historical_output('\nThere are', len(unique_doc_ids), 'unique docs in a total of', len(doc_ids), 
                           sub_directory=str(question_id), filename='question_stats.log', display_on_screen=False)
    save_historical_output('most_similar_faiss_sentences:', most_similar_faiss_sentences, 
                           sub_directory=str(question_id), filename='question_stats.log', display_on_screen=False)
    
    save_historical_output('\n\tThe doc ID, doc name and sentences used for this summary were:', 
                           sub_directory=str(question_id), filename='question_stats.log')
    for doc_id, faiss_id, sentence in most_similar_faiss_sentences:
        # Need doc ID, name and sentence
        doc_filename, document = index_processor.get_document_info(doc_id)        
        save_historical_output('\n\t', doc_id, doc_filename, '\n\t\t', sentence,
                               sub_directory=str(question_id), filename='question_stats.log')
    
    for methodology in methodologies:
        process_methodology(question_id, user_question, most_similar_faiss_sentences, methodology)
        
def process_methodology(question_id, user_question, most_similar_faiss_sentences, methodology):
    global global_score_df
    global conversation_histories
    
    save_historical_output('\n\nMETHODOLOGY:', methodology, sub_directory=str(question_id), filename='question_stats.log')
    save_historical_output(methodology_descriptions.get(methodology), sub_directory=str(question_id), filename='question_stats.log')
    save_historical_output('ORIGINAL QUESTION:', user_question, sub_directory=str(question_id), filename='question_stats.log')
    
    # What to do here???  You need to actually process the answer at some point but is this the right place?
    # Keep in mind that you may be doing post-processing, conversational QA...
    # To answer a question, you need the question and the 'context' which can be any number of things based on the 
    # methodology and conversation history.
    
    faiss_VECTOR_sentences = []
#     most_similar_faiss_sentences = []
#     if methodology not in ['Question Only', 'Question & Generated Summaries']: 
#         most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
#     most_similar_faiss_sentences, methodology_context = get_methodology_context(question_id, user_question, methodology)
#     most_similar_faiss_sentences, methodology_context = get_question_and_faiss_methodology_context(question_id, user_question, methodology)
#     save_historical_output(methodology.upper(), 'CONTEXT:', methodology_context, sub_directory=str(question_id), 
#                            filename='question_stats.log', display_on_screen=False)
#     print('most_similar_faiss_sentences:', most_similar_faiss_sentences)
#     print('\nmethodology_context:', methodology_context)
    
    # Extract doc_ids from most_similar_prompts
    # THIS SHOULD ONLY BE USED FOR DOCUMENTATION PURPOSES...  I.E., the relevant info was found in these docs...
    faiss_VECTOR_sentences = [sentence for _, _, sentence in most_similar_faiss_sentences]
    doc_ids = [doc_id for doc_id, _, _ in most_similar_faiss_sentences]
    unique_doc_ids = set(doc_ids)
    save_historical_output('\nThere are', len(unique_doc_ids), 'unique docs in a total of', len(doc_ids), 
                           sub_directory=str(question_id), filename='question_stats.log', display_on_screen=False)
    
    # TODO Generate answer arrays
    answer_array = []
    if methodology == 'Question Only':
        answer_array.append(get_llm_response(user_question, user_question))
    elif methodology == 'Question & FAISS Vector':
        methodology_context = user_question + ' ' + ' '.join(faiss_VECTOR_sentences)
        answer_array.append(get_llm_response(user_question, methodology_context))
    elif methodology == 'Question & FAISS Vector Summaries':
        faiss_context = ' ' + model_settings.summarize_text(' '.join(faiss_VECTOR_sentences), max_length=1000) + ' '
        methodology_context = user_question + ' ' + faiss_context
        answer_array.append(get_llm_response(user_question, methodology_context))
    elif methodology == 'Question & Generated Summaries':
        pass
    elif methodology == 'Question & Related Docs':
        print('Processing Question & Related Docs for', len(unique_doc_ids), 'doc IDs.')
        for doc_id in unique_doc_ids:
            doc_filename, document = index_processor.get_document_info(doc_id)
            save_historical_output('\tProcessing document:', doc_id, doc_filename,
                                   sub_directory=str(question_id), filename='question_stats.log')
            doc_filename, document = index_processor.get_document_info(doc_id)
            methodology_context = user_question + ' ' + document
            summary_answer = get_llm_response(user_question, methodology_context)
            answer_array.append(summary_answer)
            save_historical_output('\tIntermediate Summary Answer:\n\t\t', summary_answer, 
                                   sub_directory=str(question_id), filename='question_stats.log', display_on_screen=True)
    elif methodology == 'Question, FAISS Vector & Related Docs':
        user_and_faiss_context = user_question + ' ' + ' '.join(faiss_VECTOR_sentences)
        print('Processing Question, FAISS Vector & Related Docs for', len(unique_doc_ids), 'doc IDs.')
        for doc_id in unique_doc_ids:
            doc_filename, document = index_processor.get_document_info(doc_id)
            save_historical_output('\tProcessing document:', doc_id, doc_filename,
                                   sub_directory=str(question_id), filename='question_stats.log')
            doc_filename, document = index_processor.get_document_info(doc_id)
            methodology_context = user_and_faiss_context + ' ' + document
            summary_answer = get_llm_response(user_question, methodology_context)
            answer_array.append(summary_answer)
            save_historical_output('\tIntermediate Summary Answer:\n\t\t', summary_answer, 
                                   sub_directory=str(question_id), filename='question_stats.log', display_on_screen=True)
    elif methodology == 'Question, FAISS Vector Summaries & Related Docs':
        user_and_faiss_context = user_question + ' ' + model_settings.summarize_text(' '.join(faiss_VECTOR_sentences), max_length=1000)
        print('Processing Question, FAISS Vector Summaries & Related Docs for', len(unique_doc_ids), 'doc IDs.')
        for doc_id in unique_doc_ids:
            doc_filename, document = index_processor.get_document_info(doc_id)
            save_historical_output('\tProcessing document:', doc_id, doc_filename,
                                   sub_directory=str(question_id), filename='question_stats.log')
            doc_filename, document = index_processor.get_document_info(doc_id)
            methodology_context = user_and_faiss_context + ' ' + document
            summary_answer = get_llm_response(user_question, methodology_context)
            answer_array.append(summary_answer)
            save_historical_output('\tIntermediate Summary Answer:\n\t\t', summary_answer, 
                                   sub_directory=str(question_id), filename='question_stats.log', display_on_screen=True)
    elif methodology == 'Question, Generated Summaries & Related Docs':
        pass
    
    # TODO Generate answer array summary    
    merged_answers = ''
    spacer = ''
    count = 0
    for answer in answer_array:
        count += 1
        merged_answers += spacer + answer
        spacer = ' '
    
        
    final_summary = model_settings.summarize_text(merged_answers, max_length=1500)
    
    save_historical_output('\nSUMMARIZED RESPONSE:', final_summary, 
                           sub_directory=str(question_id), filename='question_stats.log')
    
#     if len(most_similar_faiss_sentences) > 0:
#         save_historical_output('\n\tThe doc ID, doc name and sentences used for this summary were:', 
#                                sub_directory=str(question_id), filename='question_stats.log')
#     for doc_id, faiss_id, sentence in most_similar_faiss_sentences:
#         # Need doc ID, name and sentence
#         doc_filename, document = index_processor.get_document_info(doc_id)        
#         save_historical_output('\n\t', doc_id, doc_filename, '\n\t\t', sentence,
#                                sub_directory=str(question_id), filename='question_stats.log')
        
        
    
    
    # PROBABLY NOT HERE...
    # Now that you have the methodology_context, you can run the query
#     summarized_response = get_llm_response(user_question, methodology_context)
#     save_historical_output('\nSUMMARIZED RESPONSE:', summarized_response, '\nEND OF', methodology.upper(), 'OUTPUT:\n', 
#                            sub_directory=str(question_id), filename='question_stats.log')
    
    
#     history_id = str(question_id) + '_' + methodology
#     conversation_history = conversation_histories.get(history_id, '')  # Do not store the tags for the history.
#     conversation_history += "<response>" + summarized_response + "</response>"
    
    # Generate random scores
    random_score1 = np.random.uniform(0, 1)
    random_score2 = np.random.uniform(0, 1)
    random_score3 = np.random.uniform(0, 1)
    average_score = (random_score1 + random_score2 + random_score3) / 3
    global_score_df.loc[len(global_score_df)] = [question_id, user_question, methodology, random_score1, random_score2, random_score3, average_score]
    save_historical_output('AVERAGE SCORE:', average_score, '\nEND OF', methodology, 'METHODOLOGY\n', 
                           sub_directory=str(question_id), filename='question_stats.log')
    

def get_llm_response(user_question, methodology_context):    
    # Get an array of answers
    decoded_outputs = model_settings.get_chunked_text_generation(methodology_context, num_return_sequences=qa_num_return_sequences)
    
#     print('DECODED OUTPUTS:')
#     for i, decoded_output in enumerate(decoded_outputs):
#         counter = str(i) + ':'
#         print('\nDECODED OUTPUT', counter, decoded_output)
    final_generated_output = " ".join(decoded_outputs)      
    # Remove the user question from the decoded text
    if user_question in final_generated_output:
        final_generated_output = final_generated_output.replace(user_question, "").strip()
    save_historical_output('\nSize of FINAL GENERATED OUTPUT is', len(final_generated_output), 
                           '\n\t', final_generated_output, display_on_screen=False)
    
    # Now build a summary of the output:
    # Summarize the generated text
    summarized_text = model_settings.summarize_text(final_generated_output, max_length=1000)
    return summarized_text

def get_question_and_faiss_methodology_context(question_id, user_question, methodology):
    faiss_context = ''
    most_similar_faiss_sentences = []
    get_faiss_vectors = False
    get_faiss_summary = False
    
    if methodology == 'Question Only':
        pass
    elif methodology == 'Question & FAISS Vector':
        get_faiss_vectors = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question & FAISS Vector Summaries':
        get_faiss_summary = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question & Generated Summaries':
        get_generated_summaries = True
    elif methodology == 'Question & Related Docs':
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question, FAISS Vector & Related Docs':
        get_faiss_vectors = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question, FAISS Vector Summaries & Related Docs':
        get_faiss_summary = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question, Generated Summaries & Related Docs':
        get_generated_summaries = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
        
    # Generate the faiss_context using sentences or sentence summaries
    if len(most_similar_faiss_sentences) > 0:
        faiss_VECTOR_sentences = [sentence for _, _, sentence in most_similar_faiss_sentences]
        if get_faiss_vectors:
            faiss_context = ' '.join(faiss_VECTOR_sentences)
        elif get_faiss_summary:
            faiss_context = ' ' + model_settings.summarize_text(' '.join(faiss_VECTOR_sentences), max_length=1000) + ' '
    
    final_context = user_question + ' ' + faiss_context
    
    return most_similar_faiss_sentences, final_context
    

def get_methodology_context(question_id, user_question, methodology):
    global conversation_histories
    global similarity_threshold
    
    instructions = ''
#     instructions = "Please use the following context.  There are five html tagged components in the context of the question.  "
#     instructions += "User questions are tagged with <user>, FAISS results and/or their summaries are tagged with <faiss>, "
#     instructions += "documents related to the FAISS results are tagged with <doc>, "
#     instructions += "model answers are tagged with <response> and conversation history is tagged with <history>."
#     instructions += "  Note that user, faiss, doc and response tags can all exist within the history.  "
#     instructions += "If you understand this, precede your answer with 'INSTRUCTIONS UNDERSTOOD' on the first line "
#     instructions += "unless you see that that response is already in a prior response or history section.  \n"
    
    # The contexts are sent to the model and contain tags
    # The total context is instructions, conversation_context, question_context, faiss_context and document_context
    conversation_context = ''                                    # Prior Q & A
    question_context = user_question                             # User's actual question
#     question_context = "\n<user>" + user_question + "</user>"
    faiss_context = ''                                           # Vector DB sentences (could be raw or summarized)
    document_context = ''                                        # Documents found via vector DB    
    
    most_similar_faiss_sentences = []
    history_id = str(question_id) + '_' + methodology
    conversation_history = conversation_histories.get(history_id, '')  # Do not store the tags for the history.
    save_historical_output('Conversation History:', conversation_history, sub_directory=str(question_id), 
                           filename='question_stats.log', display_on_screen=False) 
    if conversation_history != '':
        conversation_context = "\n" + conversation_history + " "
#         conversation_context = "\n<history>" + conversation_history + "</history>"
    else:
        # Prep the history record since it doesn't exist yet
        conversation_histories[history_id] = ''        
    
    # methodologies = ['Question Only', 'Question & FAISS Vector', 'Question & FAISS Vector Summaries', 
    #                  'Question & Generated Summaries', 'Question & Related Docs', 'Question, FAISS Vector & Related Docs', 
    #                  'Question, FAISS Vector Summaries & Related Docs', 'Question, Generated Summaries & Related Docs']
    
    # TODO: If there are FAISS vectors involved, you need to also get the doc IDs and track them somehow.
    get_faiss_vectors = False
    get_faiss_summary = False
    get_generated_summaries = False
    get_related_docs = False
    if methodology == 'Question Only':
        pass
    elif methodology == 'Question & FAISS Vector':
        get_faiss_vectors = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question & FAISS Vector Summaries':
        get_faiss_summary = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question & Generated Summaries':
        get_generated_summaries = True
    elif methodology == 'Question & Related Docs':
        get_related_docs = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question, FAISS Vector & Related Docs':
        get_related_docs = True
        get_faiss_vectors = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question, FAISS Vector Summaries & Related Docs':
        get_related_docs = True
        get_faiss_summary = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    elif methodology == 'Question, Generated Summaries & Related Docs':
        get_related_docs = True
        get_generated_summaries = True
        most_similar_faiss_sentences = index_processor.get_index_responses(user_question, similarity_threshold)
    
    # Generate the faiss_context using sentences or sentence summaries
    if len(most_similar_faiss_sentences) > 0:
        faiss_VECTOR_sentences = [sentence for _, _, sentence in most_similar_faiss_sentences]
        if get_faiss_vectors:
            faiss_context = ' '.join(faiss_VECTOR_sentences)
#             faiss_context = ''.join([f'<faiss>{s}</faiss>' for s in faiss_VECTOR_sentences])
        elif get_faiss_summary:
            faiss_context = ' ' + model_settings.summarize_text(' '.join(faiss_VECTOR_sentences), max_length=1000) + ' '
#             faiss_context = '<faiss>' + model_settings.summarize_text(' '.join(faiss_VECTOR_sentences), max_length=1000) + '</faiss>'
        
#         if get_related_docs:
#             # Extract doc_ids from most_similar_prompts
#             doc_ids = [doc_id for doc_id, _, _ in most_similar_faiss_sentences]
#             unique_doc_ids = set(doc_ids)
#             # Count occurrences of each doc_id
#             doc_id_counts = Counter(doc_ids)
#             # Find the doc_id with the highest count
#             most_common_doc_id, occurrences = doc_id_counts.most_common(1)[0]
#             generated_summaries = []
#             for doc_id in unique_doc_ids:
#                 doc_filename, document = index_processor.get_document_info(doc_id)
#                 if document is not None:
                    
            
#     unique_doc_ids = set(doc_ids)
# #     print('doc_ids', doc_ids)
#     print('unique_doc_ids', unique_doc_ids)
#     # Count occurrences of each doc_id
#     doc_id_counts = Counter(doc_ids)
#     # Find the doc_id with the highest count
#     most_common_doc_id, occurrences = doc_id_counts.most_common(1)[0]
#     ##################################################
#     ##### EXTRACT SENTENCES AND ORIGINATING DOCS #####
#     ##################################################
#     generated_summaries = []
#     for doc_id in unique_doc_ids:
#         doc_filename, document = index_processor.get_document_info(doc_id)
        
#         if document is not None:
#             print('\nPROCESSING doc_id:', doc_id, doc_filename)
#             print('\nUser question:', question)
#             summaries = process_single_large_document(conversation_history, question, faiss_sentences, document)
#             generated_summaries.extend(summaries)            
#             ##########  TESTING  ##########
#             break
    
#     # TODO: Walk through the tensors in the generated sequences, check their similarity and generate a final summary of all of them.
#     summary_count = len(generated_summaries)
#     if summary_count == 0:
#         return 'Failed to load any document summaries.'
    
#     print('Processing', summary_count, 'summaries.')
# #     print('type(generated_summaries)', type(generated_summaries), '\n', generated_summaries)
#     answer = get_answer_from_summaries(question, faiss_sentences, generated_summaries)
    
#     print('\nUser question:', question, '\nFINAL ANSWER from all summaries:\n', answer)
#     return answer
            
            
            
            
            
            
            
    
    # Append the latest information to the history record
    conversation_history += question_context + faiss_context + document_context
    conversation_histories[history_id] = conversation_history
    
    final_context = instructions + conversation_context + question_context + faiss_context + document_context
    
    save_historical_output('\nDISPLAYING OUTPUTS FOR METHODOLOGY:', methodology, sub_directory=str(question_id), 
                           filename='question_stats.log') 
    save_historical_output('User Question:', user_question, sub_directory=str(question_id), 
                           filename='question_stats.log') 
    save_historical_output('THERE ARE', len(most_similar_faiss_sentences), 'most_similar_faiss_sentences:', 
                           most_similar_faiss_sentences, sub_directory=str(question_id), filename='question_stats.log') 
    save_historical_output('faiss_context:', faiss_context, sub_directory=str(question_id), 
                           filename='question_stats.log', display_on_screen=False) 
    save_historical_output('conversation_history:', conversation_history, sub_directory=str(question_id), 
                           filename='question_stats.log', display_on_screen=False) 
    save_historical_output('document_context:', document_context, sub_directory=str(question_id), 
                           filename='question_stats.log', display_on_screen=False) 
    save_historical_output('\nFINAL CONTEXT:', final_context, '\nEND FINAL CONTEXT\n', sub_directory=str(question_id), 
                           filename='question_stats.log', display_on_screen=False) 
    
    return most_similar_faiss_sentences, final_context

# def get_most_similar_faiss_sentences(user_question):
#     global similarity_threshold
    
#     # Assume that initial question is set for now.
#     most_similar_prompts = index_processor.get_index_responses(user_question, similarity_threshold)
#     # Extract only the sentences from most_similar_prompts
#     faiss_VECTOR_sentences = [sentence for _, _, sentence in most_similar_prompts]
#     return faiss_VECTOR_sentences

def get_answer(hist_record=None, next_question=None):
    global initial_question
    global sentences_index
    global similarity_threshold
    global history_record 
    
    sentences_index = index_processor.sentences_index
    if hist_record:
        history_record = hist_record
    
    # Eight options here:
    #    - SUMMARIES:
    #        - No summaries                                                    
    #              - Q only
    #              - Q & V (just the returned vectors - no vector summary)
    #        - Summaries come from vectors                                     VS
    #        - Summaries come from LMM-generated text of vectors               GS
    #    - For each SUMMARY TYPE:
    #        - Submit question and summary to final LMM                        Q only, Q & V, Q & VS, Q & GS
    #        - Submit question, summary and relevant documents to final LMM    Q & Docs, Q, V & Docs, Q, VS, & Docs, Q, GS & Docs
    
    # Assume that initial question is set for now.
    most_similar_prompts = index_processor.get_index_responses(initial_question, similarity_threshold)
    save_historical_output('most_similar_prompts:', most_similar_prompts)
    # Extract only the sentences from most_similar_prompts
    faiss_VECTOR_sentences = [sentence for _, _, sentence in most_similar_prompts]
    
    # Extract doc_ids from most_similar_prompts
    doc_ids = [doc_id for doc_id, _, _ in most_similar_prompts]
    unique_doc_ids = set(doc_ids)
    save_historical_output('There are', len(unique_doc_ids), 'unique docs in a total of', len(doc_ids))
    
    # 2 simple cases: Q only and Q & Docs
    answer = process_inputs_and_retrieve_answer(initial_question)
    print('\nThe final question and answer was:\nQUESTION:\n\t', initial_question, '\nANSWER:\n\t', answer)
    save_historical_output('Really, really', answer, filename='answers.txt')
#     answer = process_inputs_and_retrieve_answer(initial_question, faiss_sentences=faiss_VECTOR_sentences)

    # Q and VS: Generate a summary of the vectors first and then generate the answer
#     VS = model_settings.summarize_text(' '.join(faiss_VECTOR_sentences), max_length=1000)
#     answer = process_inputs_and_retrieve_answer(initial_question, faiss_sentences=[VS])    
    
def process_inputs_and_retrieve_answer(user_question, faiss_sentences=None, documents=None):    
    # Generate the input context
    combined_input = user_question
    methodology = 'Question Only'
    
    if faiss_sentences:
        # Concatenate FAISS sentences with the user question to form the context
        combined_input += " " + ' '.join(faiss_sentences)    
        
    # Get an array of answers
    decoded_outputs = model_settings.get_chunked_text_generation(combined_input, num_return_sequences=qa_num_return_sequences)
    
#     print('DECODED OUTPUTS:')
#     for i, decoded_output in enumerate(decoded_outputs):
#         counter = str(i) + ':'
#         print('\nDECODED OUTPUT', counter, decoded_output)
    final_generated_output = " ".join(decoded_outputs)      
    # Remove the user question from the decoded text
    if user_question in final_generated_output:
        final_generated_output = final_generated_output.replace(user_question, "").strip()
    save_historical_output('\n\nSize of FINAL GENERATED OUTPUT is', len(final_generated_output), '\n\t', final_generated_output)
    
    # Now build a summary of the output:
    # Summarize the generated text
    summarized_text = model_settings.summarize_text(final_generated_output, max_length=1000)
    save_historical_output('METHODOLOGY:', methodology, filename='answers.txt')
    save_historical_output('ORIGINAL QUESTION:', user_question, filename='answers.txt')
    save_historical_output(f"SUMMARIZED TEXT: {summarized_text}", filename='answers.txt')    
    
    
def generate_question_stats(counter):
    global global_score_df
    save_historical_output('\nGenerating question statistics...', 
                           sub_directory=str(counter), filename='question_stats.log')
    
    # TODO: Get the stats for this question.  For now, this is all made up
    
    
def generate_overall_stats():
    global history_record
    global global_score_df
    stats_directory = 'overall_stats'
    save_historical_output('Generating overall statistics...', 
                           sub_directory=stats_directory, filename='overall_stats.log')
        
    # Calculate the average of each methodology's average_score
    methodology_avg_scores = global_score_df.groupby('Methodology')['average_score'].mean()
    
    # Find the highest scoring methodology
    highest_scoring_methodology = methodology_avg_scores.idxmax()
    highest_score = methodology_avg_scores.max()
    
    if history_record:
        history_record.set_tag("best_methodology", highest_scoring_methodology)
        history_record.log_metric("best_methodology", highest_scoring_methodology.replace(' ', '_'))
        history_record.log_metric("methodology_score", highest_score)
        
        # Previous logs should have created the directory already
        filename = os.path.join(history_record.artifact_directory, stats_directory, 'global_stats.csv')
        global_score_df.to_csv(filename, index=False)
        

    # Output the name and average score of the highest scoring methodology
    save_historical_output(f"Highest Scoring Methodology: {highest_scoring_methodology}", 
                           sub_directory=stats_directory, filename='overall_stats.log')
    save_historical_output(f"Average Score: {highest_score}", 
                           sub_directory=stats_directory, filename='overall_stats.log')
    
'''
    global qa_num_return_sequences
    qa_model, qa_tokenizer = model_settings.get_qa_model_and_tokenizer()

    # Concatenate FAISS sentences with the user question to form the context
    faiss_context = ' '.join(faiss_sentences)
    context = user_question + " " + faiss_context
#     context = user_question + " " + faiss_context + " " + large_document

    # Tokenize the context
    inputs = qa_tokenizer.encode(context, return_tensors="pt")

    # Set the maximum sequence length according to the model's maximum token limit
    max_length = qa_model.config.max_position_embeddings

    # Print the length of the combined context and max length
    print("Length of combined context:", len(context.split()), 'max_length:', max_length)

    # Generate text using GPT-2 model
    generated = qa_model.generate(inputs, max_length=max_length, num_return_sequences=qa_num_return_sequences, do_sample=True)
    
    # Decode and print the generated text
    summaries = []
    for i, g in enumerate(generated):
        decoded_text = qa_tokenizer.decode(g, skip_special_tokens=True)        
        # Remove the user question from the decoded text
        if user_question in decoded_text:
            decoded_text = decoded_text.replace(user_question, "").strip()
            
        print(f"\nGENERATED SEQUENCE {i + 1}: {decoded_text[:300]}")
        # Summarize the generated text
        summarized_text = model_settings.summarize_text(decoded_text, max_length=500)
        print(f"\nSUMMARIZED SEQUENCE {i + 1}: {summarized_text}\n")
        summaries.append(summarized_text)

#     return generated  
    return summaries
'''

def run_lmm_tests(hist_record=None, next_question=None):
    global initial_question
    global sentences_index
    global similarity_threshold
    global history_record
    global conversation_history
    
    sentences_index = index_processor.sentences_index
        
    if hist_record:
        history_record = hist_record
    
    if history_record:
        history_record.log_param("Docs", len(index_processor.document_guid_lookup_df))
        history_record.log_param("Idx Size", sentences_index.ntotal)
    
    # TASKS:
    # Generate question, query index, generate prompts for final LMM, get final answer
    # The Ensembles example has this in the code: if not pretrained_model_directory:
    # This can be substituted with questions and answers that have occurred in this 'conversation'
    save_historical_output('Running LMM Tests...')
        
    if next_question:
        # Can't run the next question if there wasn't a single question
        pass
    elif initial_question:
        most_similar_prompts = index_processor.get_index_responses(initial_question, similarity_threshold)
        answer = answer_query_using_all_relevant_docs(conversation_history, initial_question, most_similar_prompts)
    else:
        questions = get_questions()
        for question in questions:
            # For each question, get the list of best sentence fragment matches from the index.
            most_similar_prompts = index_processor.get_index_responses(question, similarity_threshold)

            # Now that you have the most_similar_prompts, call the method to run the query
    #         answer = answer_query(conversation_history, question, most_similar_prompts)
            answer = answer_query_using_all_relevant_docs(conversation_history, question, most_similar_prompts)
    
    # Since there can be multiple questions, you need to perform different scoring for ALL of them here.
    

def get_questions():
    questions = ["Could you provide a comprehensive overview of the impact of climate change on global health, considering various aspects and potential implications?", 
                 "How is global health affected by climate change?", 
                 "How will AI help us in the future?", 
                 "What are the main concerns with AI?"]
#     return questions[1:3]
    return questions[3:]
#     return questions

def answer_query_using_all_relevant_docs(conversation_history, question, most_similar_prompts):
    ##################################################
    ##### EXTRACT SENTENCES AND ORIGINATING DOCS #####
    ##################################################
    # Extract only the sentences from most_similar_prompts
    faiss_sentences = [sentence for _, _, sentence in most_similar_prompts]    
    # Temporary until I find a better way to question all of the relevant documents
    # Extract doc_ids from most_similar_prompts
    doc_ids = [doc_id for doc_id, _, _ in most_similar_prompts]
    unique_doc_ids = set(doc_ids)
#     print('doc_ids', doc_ids)
    print('unique_doc_ids', unique_doc_ids)
    # Count occurrences of each doc_id
    doc_id_counts = Counter(doc_ids)
    # Find the doc_id with the highest count
    most_common_doc_id, occurrences = doc_id_counts.most_common(1)[0]
    ##################################################
    ##### EXTRACT SENTENCES AND ORIGINATING DOCS #####
    ##################################################
    generated_summaries = []
    for doc_id in unique_doc_ids:
        doc_filename, document = index_processor.get_document_info(doc_id)
        
        if document is not None:
            print('\nPROCESSING doc_id:', doc_id, doc_filename)
            print('\nUser question:', question)
            summaries = process_single_large_document(conversation_history, question, faiss_sentences, document)
            generated_summaries.extend(summaries)            
            ##########  TESTING  ##########
            break
    
    # TODO: Walk through the tensors in the generated sequences, check their similarity and generate a final summary of all of them.
    summary_count = len(generated_summaries)
    if summary_count == 0:
        return 'Failed to load any document summaries.'
    
    print('Processing', summary_count, 'summaries.')
#     print('type(generated_summaries)', type(generated_summaries), '\n', generated_summaries)
    answer = get_answer_from_summaries(question, faiss_sentences, generated_summaries)
    
    print('\nUser question:', question, '\nFINAL ANSWER from all summaries:\n', answer)
    return answer
    
def process_single_large_document(conversation_history, user_question, faiss_sentences, large_document):
    global qa_num_return_sequences
    qa_model, qa_tokenizer = model_settings.get_qa_model_and_tokenizer()

    # Concatenate FAISS sentences with the user question to form the context
    faiss_context = ' '.join(faiss_sentences)
    context = user_question + " " + faiss_context
#     context = user_question + " " + faiss_context + " " + large_document

    # Tokenize the context
    inputs = qa_tokenizer.encode(context, return_tensors="pt")

    # Set the maximum sequence length according to the model's maximum token limit
    max_length = qa_model.config.max_position_embeddings

    # Print the length of the combined context and max length
    print("Length of combined context:", len(context.split()), 'max_length:', max_length)

    # Generate text using GPT-2 model
    generated = qa_model.generate(inputs, max_length=max_length, num_return_sequences=qa_num_return_sequences, do_sample=True)
    
    # Decode and print the generated text
    summaries = []
    for i, g in enumerate(generated):
        decoded_text = qa_tokenizer.decode(g, skip_special_tokens=True)        
        # Remove the user question from the decoded text
        if user_question in decoded_text:
            decoded_text = decoded_text.replace(user_question, "").strip()
            
        print(f"\nGENERATED SEQUENCE {i + 1}: {decoded_text[:300]}")
        # Summarize the generated text
        summarized_text = model_settings.summarize_text(decoded_text, max_length=500)
        print(f"\nSUMMARIZED SEQUENCE {i + 1}: {summarized_text}\n")
        summaries.append(summarized_text)

#     return generated  
    return summaries

def get_answer_from_summaries(question, faiss_sentences, generated_summaries):
    global history_record
    index_transformer_model, _ = index_processor.get_index_transformer_model_and_tokenizer()
    
    # TODO: Compare their similarities, log the metrics,generate a final summary from all of them and log that also.
    # Metrics should include cosine similarity between all pairs, a final, overall cos-sim average and the best cos-sim.
    print('Generating a final answer and statistics')
    merged_summaries = ''
    spacer = ''
    count = 0
    for generated_summary in generated_summaries:
        count += 1
#         print('Merging generated_sequence type:', type(generated_summary), '\n', generated_summary)
        merged_summaries += spacer + generated_summary
        spacer = ' '
        
    final_summary = model_settings.summarize_text(merged_summaries, max_length=500)
    # Compare the final summary to each of the generated summaries
    best_summary_to_final_score = 0.0
    average_summary_comparison_score = 0.0
    score_total = 0.0
    question_embedding = index_transformer_model.encode(question)
    final_embedding = index_transformer_model.encode(final_summary)
    question_to_final_answer_score = util.cos_sim(question_embedding, final_embedding).item()
    
    for generated_summary in generated_summaries:
        generated_summary_embedding = index_transformer_model.encode(generated_summary)
        cos_sim = util.cos_sim(generated_summary_embedding, final_embedding).item()
        score_total += cos_sim
        if history_record:
            history_record.log_metric("gen_to_final", cos_sim)
        if cos_sim > best_summary_to_final_score:
            best_summary_to_final_score = cos_sim
    
    average_summary_comparison_score = score_total / count
    print('\tThe average summary comparison score is', average_summary_comparison_score, 
          '\n\tThe best summary comparison score is', best_summary_to_final_score, 
          '\n\tThe final summary comparison to the question is', question_to_final_answer_score)
    if history_record:
        history_record.log_metric("best_to_final", best_summary_to_final_score)
        history_record.log_metric("quest_to_final", question_to_final_answer_score)
    
    return final_summary

