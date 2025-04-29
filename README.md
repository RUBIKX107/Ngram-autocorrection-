# Ngram-autocorrection-

ngram-text-prediction/
├── data/
│   └── pride_and_prejudice.txt  # Optional: Include a small sample text or link to Project Gutenberg
├── src/
│   └── ngram_predictor.py       # Main Python script with your code
├── notebooks/
│   └── ngram_autocompletion.ipynb  # Jupyter notebook version of your Colab code
├── requirements.txt             # List of dependencies
├── README.md                   # Project description
└── LICENSE                     # License file (e.g., MIT)


# N-gram Text Autocompletion

A Python-based project to predict the next word in a sentence using bigram and trigram models, trained on *Pride and Prejudice* from Project Gutenberg. Built as a mini-project after completing Brilliant's course on language models, this demonstrates n-gram concepts applied to text prediction, like those used in smartphone keyboards and chatbots.

## Features
- Predicts the next word for a given word (bigram) or two-word phrase (trigram).
- Trained on *Pride and Prejudice* text corpus.
- Visualizes top predictions using matplotlib.
- Interactive testing with user inputs.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ngram-text-prediction.git
   cd ngram-text-prediction

## Usage 
   pip install -r requirements.txt

   # Step 1: Import Libraries
import re
import requests
from collections import defaultdict, Counter
import random

# Step 2: Load and Preprocess Text Corpus
# We'll use a sample text from Project Gutenberg (e.g., Pride and Prejudice)
url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = requests.get(url)
text = response.text

# Clean the text: lowercase, remove punctuation, and split into words
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    return words

words = preprocess_text(text)
print("Sample words:", words[:10])

# Step 3: Build Bigram Model
# Create a dictionary to store bigrams and their counts
bigrams = defaultdict(Counter)
for i in range(len(words) - 1):
    bigrams[words[i]][words[i + 1]] += 1

# Step 4: Calculate Probabilities
# Convert counts to probabilities for each bigram
bigram_probs = defaultdict(dict)
for word1 in bigrams:
    total_count = sum(bigrams[word1].values())
    for word2 in bigrams[word1]:
        bigram_probs[word1][word2] = bigrams[word1][word2] / total_count

# Step 5: Autocompletion Function
def predict_next_word(current_word, ngram_probs):
    if current_word not in ngram_probs:
        return "Unknown word"
    # Get the next word probabilities
    next_words = ngram_probs[current_word]
    if not next_words:
        return "No prediction"
    # Choose the word with the highest probability
    predicted_word = max(next_words, key=next_words.get)
    return predicted_word

# Step 6: Test the Model
test_words = ["the", "and", "mr"]
for word in test_words:
    next_word = predict_next_word(word, bigram_probs)
    print(f"Input: {word} -> Predicted next word: {next_word}")

# Bonus: Extend to Trigrams (Optional)
# Create trigram model
trigrams = defaultdict(Counter)
for i in range(len(words) - 2):
    trigram_key = (words[i], words[i + 1])
    trigrams[trigram_key][words[i + 2]] += 1

# Convert trigram counts to probabilities
trigram_probs = defaultdict(dict)
for key in trigrams:
    total_count = sum(trigrams[key].values())
    for word in trigrams[key]:
        trigram_probs[key][word] = trigrams[key][word] / total_count

# Trigram prediction function
def predict_next_word_trigram(word1, word2, ngram_probs):
    key = (word1, word2)
    if key not in ngram_probs:
        return "Unknown sequence"
    next_words = ngram_probs[key]
    if not next_words:
        return "No prediction"
    predicted_word = max(next_words, key=next_words.get)
    return predicted_word

# Test trigram model
test_pairs = [("the", "day"), ("mr", "darcy"), ("and", "the")]
for word1, word2 in test_pairs:
    next_word = predict_next_word_trigram(word1, word2, trigram_probs)
    print(f"Input: {word1} {word2} -> Predicted next word: {next_word}")


    
   
