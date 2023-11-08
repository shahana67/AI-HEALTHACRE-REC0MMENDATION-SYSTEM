# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a list of common English stop words
stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't"
]

# Load your medicine data
df = pd.read_csv('medicine.csv')

# Remove rows with missing values
df.dropna(inplace=True)

# Combine 'Description' and 'Reason' columns
df['tags'] = df['Description'] + df['Reason']

# Create a CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words=stop_words)

# Fit and transform the text data
vectors = cv.fit_transform(df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# Function to recommend medicines based on patient symptoms
def recommend_medicine(symptoms):
    query_vector = cv.transform([symptoms]).toarray()

    # Calculate similarity between the query vector and all medicines
    similarities = cosine_similarity(vectors, query_vector)

    # Get the top 5 similar medicines
    top_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:5]

    recommended_medicines = [df.iloc[i[0]]['Drug_Name'] for i in top_similarities]

    return recommended_medicines

# User input for symptoms
patient_symptoms = input("Enter the symptoms or disease: ")
recommended_medicines = recommend_medicine(patient_symptoms)

print(f"Recommended medicines for a patient with symptoms '{patient_symptoms}':")
for med in recommended_medicines:
    print(med)
