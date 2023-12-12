
# TODO: Figure out a way to track/update the various models and tokenizers
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertForQuestionAnswering


index_transformer_model = None
index_transformer_model_name = 'sentence-transformers/all-mpnet-base-v2'
index_transformer_tokenizer = None

summarization_model = None
summarization_model_name = 'facebook/bart-large-cnn'
summarization_tokenizer = None

qa_model = None
qa_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
qa_tokenizer = None

# For now, just provide some hard-coded model retrieval methods
def get_index_transformer_model_and_tokenizer():
    global index_transformer_model
    global index_transformer_tokenizer
    if index_transformer_model is None:
        # Load it
        index_transformer_model = SentenceTransformer(index_transformer_model_name)
    return index_transformer_model, index_transformer_tokenizer
        
def get_summarization_model_and_tokenizer():
    global summarization_model
    global summarization_model_name
    global summarization_tokenizer
    
    if summarization_model is None or summarization_tokenizer is None:
        # Load the BART tokenizer
        summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)

        # Load the pre-trained BART model for summarization
        summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

    return summarization_model, summarization_tokenizer
        
def get_qa_model_and_tokenizer():
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
