from flask import Flask, render_template, request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample ratings: Items = Movies (rows), Users = Columns
ratings = {
    'User1': {'MovieA': 5, 'MovieB': 3, 'MovieC': 4, 'MovieE': 2},
    'User2': {'MovieA': 4, 'MovieB': 5, 'MovieD': 3, 'MovieE': 2},
    'User3': {'MovieA': 4, 'MovieC': 5, 'MovieD': 4},
    'User4': {'MovieB': 3, 'MovieC': 4, 'MovieD': 5, 'MovieE': 3},
}

users = list(ratings.keys())
movies = sorted({movie for r in ratings.values() for movie in r})

def cosine_sim(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a) and np.linalg.norm(b) else 0

def get_user_vector(user, items):
    return [ratings[user].get(item, 0) for item in items]

def get_item_vector(item, user_list):
    return [ratings[user].get(item, 0) for user in user_list]

def predict_user_based(user, item):
    numerator = denominator = 0
    target_vec = get_user_vector(user, movies)
    for other in users:
        if other == user or item not in ratings[other]: continue
        sim = cosine_sim(target_vec, get_user_vector(other, movies))
        numerator += sim * ratings[other][item]
        denominator += sim
    return numerator / denominator if denominator else None

def predict_item_based(user, item):
    if item in ratings[user]: return ratings[user][item]
    numerator = denominator = 0
    for other_item, rating in ratings[user].items():
        if other_item == item: continue
        sim = cosine_sim(get_item_vector(item, users), get_item_vector(other_item, users))
        numerator += sim * rating
        denominator += sim
    return numerator / denominator if denominator else None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user = request.form['user']
        movie = request.form['movie']
        user_based = predict_user_based(user, movie)
        item_based = predict_item_based(user, movie)
        prediction = {'user_based': user_based or 0, 'item_based': item_based or 0}
    return render_template('index.html', users=users, movies=movies, prediction=prediction, ratings=ratings)

if __name__ == '__main__':
    app.run(debug=True)
