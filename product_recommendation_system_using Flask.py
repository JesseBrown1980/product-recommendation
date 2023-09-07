from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('ratings.csv')

user_item_matrix = data.pivot(index='user_id', columns='product_id', values='interaction_score')
user_item_matrix = user_item_matrix.fillna(0)

user_similarity = cosine_similarity(user_item_matrix)

product_profile_matrix = data.pivot_table(index='product_id', columns='category', values='interaction_score', fill_value=0)

def get_personalized_rankings(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_index = user_item_matrix.index.get_loc(user_id)

    collaborative_scores = user_similarity[user_index] @ user_item_matrix.values
    collaborative_ranking = list(user_item_matrix.columns[np.argsort(-collaborative_scores)])[:top_n]

    content_scores = user_item_matrix.loc[user_id] @ product_profile_matrix.values
    content_ranking = list(product_profile_matrix.index[np.argsort(-content_scores)])[:top_n]

    hybrid_ranking = collaborative_ranking + [p for p in content_ranking if p not in collaborative_ranking]
    return hybrid_ranking[:top_n]  # Return only the top 5 products

def get_top_10_products_from_all_users():
    all_products = data['product_id'].unique()
    product_scores = {}

    for product_id in all_products:
        product_scores[product_id] = 0

    for user_id in user_item_matrix.index:
        personalized_rankings = get_personalized_rankings(user_id)
        for rank, product_id in enumerate(personalized_rankings, start=1):
            product_scores[product_id] += (len(personalized_rankings) - rank + 1)

    top_10_products = sorted(product_scores, key=lambda x: product_scores[x], reverse=True)[:10]
    return top_10_products

@app.route('/')
def index():
    user_ids = data['user_id'].unique()[:5]  # Limit to 5 user IDs
    personalized_rankings_dict = {}
    top_10_products_all_users = get_top_10_products_from_all_users()

    for user_id in user_ids:
        personalized_rankings = get_personalized_rankings(user_id, top_n=5)  # Limit to top 5 products per user
        personalized_rankings_dict[user_id] = []

        for product_id in personalized_rankings:
            product_name = data[data['product_id'] == product_id]['product_name'].values[0]
            image_link = data[data['product_id'] == product_id]['Image Link'].values[0]
            personalized_rankings_dict[user_id].append({
                'product_name': product_name,
                'image_link': image_link
            })

    top_10_products_info = []
    for product_id in top_10_products_all_users:
        product_name = data[data['product_id'] == product_id]['product_name'].values[0]
        image_link = data[data['product_id'] == product_id]['Image Link'].values[0]
        top_10_products_info.append({
            'product_name': product_name,
            'image_link': image_link
        })

    return render_template('algo.html', rankings_dict=personalized_rankings_dict, top_10_products=top_10_products_info)

if __name__ == '__main__':
    app.run(debug=True)
