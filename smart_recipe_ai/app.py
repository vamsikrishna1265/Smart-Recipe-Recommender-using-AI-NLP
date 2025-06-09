from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("indian_food_recipes.csv")

# Normalize column names: strip spaces, lowercase, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert cooking_time_minutes column to numeric, drop rows where invalid or missing
df['cooking_time_minutes'] = pd.to_numeric(df['cooking_time_minutes'], errors='coerce')
df.dropna(subset=['cooking_time_minutes'], inplace=True)
df['cooking_time_minutes'] = df['cooking_time_minutes'].astype(int)

# Prepare TF-IDF vectorizer on ingredients column
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_ingredients = request.form['ingredients']
    user_time = request.form['time']

    # Validate time input is a number
    try:
        user_time = int(user_time)
    except ValueError:
        return render_template('index.html', result="⛔ Please enter a valid cooking time (in minutes).")

    # Convert input ingredients to TF-IDF vector
    user_vec = vectorizer.transform([user_ingredients])

    # Filter recipes by cooking time
    df_filtered = df[df['cooking_time_minutes'] <= user_time].copy()

    if df_filtered.empty:
        return render_template('index.html', result="⚠️ No recipes found within the given cooking time.")

    # Compute cosine similarity between user input and filtered recipes
    similarity = cosine_similarity(user_vec, vectorizer.transform(df_filtered['ingredients']))

    # Find best matching recipe index
    best_idx = similarity.argmax()
    best_recipe = df_filtered.iloc[best_idx]

    # Prepare output
    output = f"""
    ✅ Recipe: <b>{best_recipe['recipe_name']}</b><br>
    ⏱️ Cooking Time: {best_recipe['cooking_time_minutes']} minutes<br>
    📝 Instructions:<br>{best_recipe['instructions']}
    """

    return render_template('index.html', result=output)

if __name__ == '__main__':
    app.run(debug=True)
