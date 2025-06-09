import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Load dataset
df = pd.read_csv('indian_food_recipes.csv')
df.columns = [col.strip() for col in df.columns]
df['cooking_time_minutes'] = pd.to_numeric(df['cooking_time_minutes'], errors='coerce')
df.dropna(subset=['ingredients', 'cooking_time_minutes'], inplace=True)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

# Simulate realistic user input: partial and shuffled ingredients
def simulate_user_input(ingredients):
    items = [item.strip() for item in ingredients.split(',') if item.strip()]
    if len(items) <= 3:
        return ', '.join(items)  # not enough to reduce
    random.shuffle(items)
    keep_n = random.randint(3, min(6, len(items)))  # randomly keep 3–6 ingredients
    return ', '.join(items[:keep_n])

# Evaluate the system
def evaluate_model(top_k_list=[1, 3, 5], sample_size=50):
    top_hits = {k: 0 for k in top_k_list}
    mrr_total = 0
    test_samples = df.sample(sample_size, random_state=42)

    for idx, row in test_samples.iterrows():
        user_input = simulate_user_input(row['ingredients'])
        actual_name = row['recipe_name']

        # Vectorize user input
        user_vec = vectorizer.transform([user_input])
        sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
        sorted_indices = sim_scores.argsort()[::-1]

        recommended_names = df.iloc[sorted_indices]['recipe_name'].values.tolist()

        # Top-K accuracy
        for k in top_k_list:
            if actual_name in recommended_names[:k]:
                top_hits[k] += 1

        # MRR
        if actual_name in recommended_names:
            rank = recommended_names.index(actual_name) + 1
            mrr_total += 1 / rank

    print("\n📊 Evaluation Results (with simulated user input):")
    for k in top_k_list:
        acc = top_hits[k] / sample_size
        print(f"Top-{k} Accuracy: {acc:.2f} ({top_hits[k]}/{sample_size})")

    mrr = mrr_total / sample_size
    print(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")

# Run evaluation
if __name__ == '__main__':
    evaluate_model()
