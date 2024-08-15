import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load("en_core_web_md")

resumes = [
    "Senior Software Developer specializing in Java and Python with extensive experience in backend systems.",
    "Project Manager with a decade of experience in successfully managing large construction projects.",
    "Marketing Specialist focused on digital advertising and social media campaigns, especially in tech startups.",
    "Lead Data Scientist with a strong background in machine learning and Python in financial services.",
    "Senior Network Engineer managing large-scale enterprise networks with a focus on security and efficiency.",
    "Digital Marketing Manager with a strong track record in content creation and lead generation in B2B markets.",
    "Human Resources Manager with extensive experience in tech industry recruitment and employee management.",
    "Product Manager overseeing the complete lifecycle of consumer electronics products, from concept to market.",
    "Experienced Sales Executive in pharmaceuticals with a strong history of exceeding sales targets.",
    "Financial Analyst with deep insights into equity research and investment strategies for technology firms.",
    "Software Engineer with a focus on mobile applications using Swift and Kotlin.",
    "Business Analyst with a knack for process improvements and cost-saving strategies in manufacturing.",
    "Database Administrator specializing in SQL Server management for large healthcare systems.",
    "Public Relations Manager who has handled crisis communications for large public events.",
    "Investment Banker focusing on mergers and acquisitions in the tech sector."
]

def preprocess_resumes(resumes):
    preprocessed_resumes = []
    for resume in resumes:
        doc = nlp(resume)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        preprocessed_resumes.append(" ".join(tokens))
    return preprocessed_resumes

preprocessed_resumes = preprocess_resumes(resumes)

tfidf_vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_resumes)

silhouette_scores = []
for k in range(2, min(len(resumes), 6)):  
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    score = silhouette_score(tfidf_matrix, clusters)
    silhouette_scores.append((k, score))
    print(f'Silhouette Score for {k} clusters: {score}')

optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f'Optimal number of clusters: {optimal_k}')

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('PCA Cluster Visualization')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar()
plt.show()

def get_top_features_cluster(tfidf_matrix, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label)  
        x_means = np.mean(tfidf_matrix[id_temp], axis=0)  
        sorted_means = np.argsort(x_means)[::-1][:n_feats]  
        features = tfidf_vectorizer.get_feature_names_out()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df)
    return dfs

top_features_per_cluster = get_top_features_cluster(tfidf_matrix, clusters, 5)
for num, df in enumerate(top_features_per_cluster):
    print(f"Top features for cluster {num}:")
    print(df)
