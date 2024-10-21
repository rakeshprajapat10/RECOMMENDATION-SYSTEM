import pandas as pd
from surprise import KNNWithMeans, Dataset, accuracy
from surprise.model_selection import train_test_split
ratings = pd.read_csv('u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
ratings = ratings[['user_id', 'item_id', 'rating']]
data = Dataset.load_from_df(ratings, rating_scale=(1, 5))
trainset, testset = train_test_split(data, test_size=.25)
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})

algo.fit(trainset)

predictions = algo.test(testset)

accuracy.rmse(predictions)
accuracy.mae(predictions)

from collections import defaultdict

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

test = algo.test(trainset.build_testset())
top_n = get_top_n(test, n=10)

for uid, user_ratings in top_n.items():
    print("User ID: ", uid)
    for iid, est in user_ratings:
        print("Item ID: ", iid, "Estimated rating: ", est)