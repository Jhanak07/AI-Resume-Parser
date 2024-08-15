import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load the medium-sized English model with word embeddings
nlp = spacy.load("en_core_web_md")

# Define a list of words from the resumes for which you want to calculate similarity
resume_words = ["experience", "skills", "education", "communication", "programming", "leadership"]

# Calculate the semantic similarity between each pair of words
similarity_matrix = [[nlp(word1).similarity(nlp(word2)) for word2 in resume_words] for word1 in resume_words]

# Set the size of the heatmap
plt.figure(figsize=(8, 6))

# Create the heatmap using seaborn
sns.heatmap(similarity_matrix, annot=True, xticklabels=resume_words, yticklabels=resume_words, cmap="YlGnBu")

# Set the title and labels
plt.title("Semantic Similarity Heatmap of Resume Words")
plt.xlabel("Resume Words")
plt.ylabel("Resume Words")

# Display the heatmap
plt.show()
