import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

resume_sentence = "Experienced software engineer proficient in Python and Java."

tokens = [token.text for token in nlp(resume_sentence)]
print("Step 1: Tokenization")
print(tokens)
print()

tokens_no_stopwords = [token for token in tokens if token.lower() not in STOP_WORDS]
print("Step 2: Stop Word Removal")
print(tokens_no_stopwords)
print()

lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens_no_stopwords))]
print("Step 3: Lemmatization")
print(lemmatized_tokens)
