
# TODO: Figure out a way to track/update the various models and tokenizers
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertForQuestionAnswering
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import math
import torch

# Set the logging level to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

summarization_model = None
summarization_model_name = 'facebook/bart-large-cnn'
summarization_tokenizer = None

qa_model = None
qa_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
qa_tokenizer = None
        
def get_summarization_model_and_tokenizer():
    global summarization_model
    global summarization_model_name
    global summarization_tokenizer
    
    if summarization_model is None or summarization_tokenizer is None:
        print('LOADING the summarization model and tokenizer...')
        # Load the BART tokenizer
        summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)

        # Load the pre-trained BART model for summarization
        summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)
    else:
        print('USING THE PRELOADED summarization model and tokenizer...')

    return summarization_model, summarization_tokenizer

# Allow the model and tokenizer to be stored on Jupyter and pushed into memory instead of reloading it.
def set_summarization_model_and_tokenizer(model, tokenizer):
    global summarization_model
    global summarization_model_name
    global summarization_tokenizer
    print('SETTING the summarization model and tokenizer...')
    summarization_model = model
#     summarization_model_name = model_name
    summarization_tokenizer = tokenizer    
        
def get_qa_model_and_tokenizer():
    global qa_model
    global qa_model_name
    global qa_tokenizer
    
    if qa_model is None or qa_tokenizer is None:
        print('LOADING the QA model and tokenizer...')
        qa_model_name = 'gpt2'
        qa_model = GPT2LMHeadModel.from_pretrained(qa_model_name)
        qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_name)
        # Set padding token
        qa_tokenizer.pad_token = qa_tokenizer.eos_token
#         # Load the BART tokenizer
#         qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

#         # Load the pre-trained BERT model for question answering
#         qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)

    return qa_model, qa_tokenizer

# Allow the model and tokenizer to be stored on Jupyter and pushed into memory instead of reloading it.
def set_qa_model_and_tokenizer(model, tokenizer):
    global qa_model
    global qa_model_name
    global qa_tokenizer
    print('SETTING the QA model and tokenizer...')
    qa_model = model
#     qa_model_name = model_name
    qa_tokenizer = tokenizer

def get_qa_model_and_tokenizer_with_BERT():
    global qa_model
    global qa_model_name
    global qa_tokenizer
    
    if qa_model is None or qa_tokenizer is None:
        # Load the BART tokenizer
        qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

        # Load the pre-trained BERT model for question answering
        qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)

    return qa_model, qa_tokenizer

def summarize_text(text, max_length=150):
    global summarization_model
    global summarization_tokenizer
    if summarization_model is None or summarization_tokenizer is None:
        get_summarization_model_and_tokenizer()
        
    # Tokenize the input text
    inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = summarization_model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary tokens into text
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    
# TODO: Modify this to return an array of answers for the cases where you want multiple answers.
def get_chunked_text_generation(combined_input, num_return_sequences=1):
    global qa_model
    global qa_tokenizer
    if qa_model is None or qa_tokenizer is None:
        qa_model, qa_tokenizer = get_qa_model_and_tokenizer()
    
    # Set the maximum sequence length according to the model's maximum token limit
    max_length = qa_model.config.max_position_embeddings
    
    # Split the combined input into chunks
    chunk_size = 1000  # Define the maximum chunk size
    num_chunks = math.ceil(len(combined_input) / chunk_size)
    tokenized_chunks = []

    print('\tProcessing', num_chunks, 'text chunk(s).')
    if num_chunks > 6:
        print('\tNumber of chunks is too large for input.  Summarizing input and chunking that...')
        # Large docs will have to be summarized first
        combined_input = summarize_text(combined_input, max_length=2000)
        # Split the combined input into chunks
        chunk_size = 1000  # Define the maximum chunk size
        num_chunks = math.ceil(len(combined_input) / chunk_size)
        tokenized_chunks = []
        print('\tNow processing', num_chunks, 'text chunk(s).')
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = combined_input[start_idx:end_idx]
        tokenized_chunk = qa_tokenizer.encode(chunk, return_tensors="pt")
        tokenized_chunks.append(tokenized_chunk)

    # Generate text using GPT-2 language model for each tokenized chunk
    generated_outputs = []

    print('\tProcessing tokenized chunk(s).')
    for chunk in tokenized_chunks:
        print('.', end='')
        # Ensure attention mask is set
        attention_mask = torch.ones_like(chunk)

        generated_output = qa_model.generate(
            chunk,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=qa_tokenizer.eos_token_id,  # Set pad token ID
            do_sample=True
        )
        generated_outputs.append(generated_output)
    print(' - COMPLETED...')

    # Decode and concatenate the generated outputs
#     decoded_outputs = [qa_tokenizer.decode(output[0], skip_special_tokens=True) for output in generated_outputs]
    # Decode the generated outputs
    decoded_outputs = [
        [qa_tokenizer.decode(output[idx], skip_special_tokens=True) for idx in range(num_return_sequences)]
        for output in generated_outputs
    ]
    # Flatten the decoded outputs to a single list of answers
    flattened_outputs = [item for sublist in decoded_outputs for item in sublist]
    
#     print('\n\ndecoded_outputs:\n\t', decoded_outputs, '\n\n')
    return flattened_outputs



