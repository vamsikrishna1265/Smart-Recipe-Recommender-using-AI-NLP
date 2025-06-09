# Smart-Recipe-Recommender-using-AI-NLP
AI-powered recipe recommender that suggests Indian dishes based on available ingredients and cooking time. Uses NLP and ML techniques to match user input with recipes, providing step-by-step cooking instructions via a Flask web app for personalized meal planning

Smart Recipe Recommender using AI & NLP
Welcome to the Smart Recipe Recommender – an AI-powered web application that suggests personalized Indian recipes based on the ingredients you currently have (like vegetables, fruits, spices) and available cooking time .

This project combines Natural Language Processing (NLP) and Machine Learning (ML) to intelligently match user inputs with the best-fitting recipes from a curated dataset of 400 unique Indian dishes. The system also provides step-by-step cooking instructions.

Project Highlights:
 AI/ML-based recipe prediction using TF-IDF and cosine similarity

 400+ authentic Indian recipes with ingredients, cook time, and instructions

 Content-based recommendation engine

 Built using Python, Flask, Pandas, and scikit-learn

 Easy-to-use web interface for live user interaction

 Includes evaluation metrics like Top-K Accuracy and MRR

Structure:
app.py – Flask backend with AI logic

templates/index.html – Frontend UI

recipes.csv – Custom Indian recipe dataset

static/ – CSS styling files

Getting Started:
Clone this repo

Install dependencies (pip install -r requirements.txt)

Run the Flask app:

python app.py
Open your browser at http://127.0.0.1:5000/ and try it live!

Use Case:
This system helps users (especially beginners or time-constrained individuals) discover recipes they can cook quickly and easily with what they already have. It has potential applications in smart kitchen assistants, mobile cooking apps, and AI-enabled meal planning.
