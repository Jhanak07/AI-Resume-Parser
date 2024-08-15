import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

nlp = spacy.load("en_core_web_md")

resume_words = [
    "Python", "Java", "C++", "HTML", "CSS", "JavaScript", "SQL", "R", "Excel",
    "TensorFlow", "scikit-learn", "AWS", "Azure", "Google Cloud", "encryption", "firewall",
    "agile", "version control", "communication", "teamwork", "leadership", "problem-solving",
    "creativity", "adaptability", "time management", "Software Engineer", "Mechanical Engineer",
    "Business Analyst", "Data Analyst", "Project Manager", "Product Manager", "Consultant",
    "Technician", "Graphic Designer", "UX/UI Designer", "PMP", "CFA", "CompTIA", "CCNA", "CCNP",
    "MCSE", "MCTS", "derivatives", "asset management", "compliance", "HIPAA", "patient care",
    "medical billing", "lean manufacturing", "Six Sigma", "CAD"
]

word_embeddings = np.array([nlp(word).vector for word in resume_words])

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(word_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='b')
for i, word in enumerate(resume_words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.title('t-SNE Visualization of Comprehensive Resume Words')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

pca = PCA(n_components=2, random_state=42)
embeddings_2d_pca = pca.fit_transform(word_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], c='r')
for i, word in enumerate(resume_words):
    plt.annotate(word, (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]))
plt.title('PCA Visualization of Comprehensive Resume Words')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
