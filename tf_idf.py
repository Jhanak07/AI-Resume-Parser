import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

resumes = [
    "Data Analyst with 3 years of experience in financial services, proficient in SQL, Python, and Tableau. Expertise in risk analysis and forecasting.",
    "Project Manager certified in PMP with 5 years of experience in healthcare projects. Strong leadership skills and expertise in Agile methodologies.",
    "Software Engineer with 10 years of experience in software development, specializing in Java and C++. Strong background in data structures and algorithms.",
    "Junior Data Analyst in retail industry, skilled in using Excel and SPSS for sales data analysis and customer behavior modeling.",
    "Senior Project Manager in IT with over 12 years of experience, specializing in managing large scale software development projects using Scrum.",
    "Software Developer with 3 years of experience in web applications, proficient in JavaScript, HTML, CSS, and React.",
    "Data Analyst with experience in the marketing sector, expertise in Google Analytics, A/B testing, and statistical analysis.",
    "Project Manager with 8 years of experience in construction, strong skills in budget management and strategic planning, certified in Lean Six Sigma.",
    "Software Engineer with 5 years of experience in mobile app development, expert in Swift and Kotlin, with a focus on iOS and Android platforms.",
    "Lead Data Analyst with 10 years of experience in healthcare, specializing in medical data analysis with advanced knowledge of SAS and Python."
]


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1, 3), stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(resumes)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("TF-IDF Matrix:")
print(tfidf_df)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

resumes_with_clusters = pd.DataFrame({"Resume": resumes, "Cluster": clusters})

print("\nResumes with Clusters:")
print(resumes_with_clusters)

def plot_clusters(data, clusters):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
    plt.title('Cluster Visualization of Resumes')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.colorbar(scatter)
    plt.show()

plot_clusters(tfidf_matrix.toarray(), clusters)
