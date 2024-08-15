import spacy
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("en_core_web_md")

resume_words = ["experience", "skills", "education", "communication", "programming", "leadership"]

similarity_matrix = [[nlp(word1).similarity(nlp(word2)) for word2 in resume_words] for word1 in resume_words]

plt.figure(figsize=(8, 6))

sns.heatmap(similarity_matrix, annot=True, xticklabels=resume_words, yticklabels=resume_words, cmap="YlGnBu")

plt.title("Semantic Similarity Heatmap of Resume Words")
plt.xlabel("Resume Words")
plt.ylabel("Resume Words")

plt.show()
