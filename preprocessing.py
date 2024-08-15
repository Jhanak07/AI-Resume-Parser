import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a sample resume sentence
resume_sentence = "Experienced software engineer proficient in Python and Java."

# Step 1: Tokenization
tokens = [token.text for token in nlp(resume_sentence)]
print("Step 1: Tokenization")
print(tokens)
print()

# Step 2: Stop Word Removal
tokens_no_stopwords = [token for token in tokens if token.lower() not in STOP_WORDS]
print("Step 2: Stop Word Removal")
print(tokens_no_stopwords)
print()

# Step 3: Lemmatization
lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens_no_stopwords))]
print("Step 3: Lemmatization")
print(lemmatized_tokens)
