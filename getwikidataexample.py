import numpy as np
import pandas as pd
import os
from datasets import load_dataset
import sys
import nltk
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from datasets import load_dataset
from gensim.models import Word2Vec
import re

embedding_dim = 64

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option('future.no_silent_downcasting', True)

# Configure stdout for UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")


# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please run: nltk.download('punkt_tab') and nltk.download('stopwords') manually.")
    sys.exit(1)
def preprocess_text(text, unique_tokens=None):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    # Add to unique tokens set if provided
    if unique_tokens is not None:
        unique_tokens.update(tokens)
    return tokens


# dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
# dataset = dataset['train'].select(range(50))

def load_wikitext_hf():
    # Load WikiText-103-raw-v1 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    #if changing this number then delete old model......... 
    dataset = dataset['train'].select(range(10_000))
    #.....................................................
    sentences = []
    unique_tokens = set()
    # Process the 'train' split (or modify for 'validation'/'test' if needed)
    for example in dataset:
        text = example['text']
        if text.strip() and not text.startswith(' ='):
            tokens = preprocess_text(text, unique_tokens)
            if tokens:
                sentences.append(tokens)
    return sentences, unique_tokens

def load_or_train_word2vec(sentences, model_path="karpathyadjusttests/wikitext_word2vec.model", vector_size=embedding_dim, window=5, min_count=5, workers=4):
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing Word2Vec model from {model_path}...")
        try:
            model = Word2Vec.load(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead...")
    
    # Train new model if none exists or loading failed
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    model.save(model_path)
    print(f"Model saved as '{model_path}'")
    return model


sentences, unique_tokens = load_wikitext_hf()
# print(sentences[:5])

model_path = "karpathyadjusttests/wikitext_word2vec.model"
print("Loading and preprocessing WikiText-103-raw-v1 data...")
if not os.path.exists(model_path):
    print("Loading and preprocessing WikiText-103-raw-v1 data...")
    sentences, unique_tokens = load_wikitext_hf()
    print(f"Number of unique tokens in the dataset: {len(unique_tokens)}")
else:
    print("Existing model found, skipping data preprocessing.")
    sentences = None
    unique_tokens = set()  # Empty set since we don't need tokens for loading
    
    # Optionally, load unique tokens from the model's vocabulary
    try:
        model = Word2Vec.load(model_path)
        unique_tokens = set(model.wv.key_to_index.keys())
        print(f"Number of unique tokens in the model's vocabulary: {len(unique_tokens)}")
    except Exception as e:
        print(f"Error loading model for vocabulary: {e}")
        print("Proceeding without vocabulary count.")
    
    # Load or train the model
model = load_or_train_word2vec(sentences, model_path)
    
vocab_size = len(unique_tokens)


sample_word = "example"
if sample_word in model.wv:
    print(f"Embedding for '{sample_word}':")
    print(model.wv[sample_word])
else:
    print(f"Word '{sample_word}' not in vocabulary.")




def get_batch(split):
    #generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
    x,y = x.to(device), y.to(device)
    return x,y