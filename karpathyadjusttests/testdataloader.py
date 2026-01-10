# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from datasets import load_dataset
# import sys
# sys.stdout.reconfigure(encoding="utf-8")

# text = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
# sample = text['train'].select(range(10))
# # for i in range(10):
# #     print(f"Row {i}:", sample[i]["text"])
# # print(sample["text"])
# # df = sample.to_pandas()
# # print(df)

# rows = sample["text"]

# # Join into one long string (with spaces or newlines)
# long_string = " ".join(rows)  
# print(long_string)


import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
from gensim.models import Word2Vec
import re

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
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

def load_wikitext_hf():
    # Load WikiText-103-raw-v1 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    dataset = dataset['train'].select(range(10_000))
    sentences = []
    # Process the 'train' split (or modify for 'validation'/'test' if needed)
    for example in dataset:
        text = example['text']
        if text.strip() and not text.startswith(' ='):
            tokens = preprocess_text(text)
            if tokens:
                sentences.append(tokens)
    return sentences

def train_word2vec(sentences, vector_size=64, window=5, min_count=5, workers=4):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model

def main():
    print("Loading and preprocessing WikiText-103-raw-v1 data...")
    sentences = load_wikitext_hf()
    print('success')
    print("Training Word2Vec model...")
    model = train_word2vec(sentences)
    
    # Save the model
    model.save("wikitext_word2vec.model")
    print("Model saved as 'wikitext_word2vec.model'")

    # Example: Get embedding for a word
    sample_word = "example"
    if sample_word in model.wv:
        print(f"Embedding for '{sample_word}':")
        print(model.wv[sample_word])
    else:
        print(f"Word '{sample_word}' not in vocabulary.")

if __name__ == "__main__":
    main()