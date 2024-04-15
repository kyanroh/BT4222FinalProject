
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy

# # Data Import and Exploration

df = pd.read_csv('./recommendations_with_score.csv')
users = pd.read_csv('./sample_user_data.csv')
games = pd.read_csv('./games_cleaned.csv')
df_with_clusters = pd.read_csv("clustering.csv")

df.head()

df.info()

users.head()

users.info()

games.head()

games.info()

# # Filtering and sampling

filtered_user_ids = users['user_id'].tolist()

df = df[df['user_id'].isin(filtered_user_ids)]
## Creating user-games matrix and calculating cosine similarity

df_recommender = df[['user_id', 'app_id', 'is_recommended', 'recommendation_credibility_normalized_log']]

df_recommender['recommendation_score'] = df_recommender['is_recommended'] * df_recommender['recommendation_credibility_normalized_log']

matrix = df_recommender.pivot_table(index='user_id', columns='app_id', values='recommendation_score', fill_value=0)

matrix.info()

user_similarity_cosine = cosine_similarity(matrix.fillna(0))
user_similarity_cosine_df = pd.DataFrame(user_similarity_cosine, index=matrix.index, columns=matrix.index)
user_similarity_cosine_df

# # Getting similar users, and the games that they played

def get_similar_users(picked_userid):
    user_similarity_threshold = 0.3
    similar_users = user_similarity_cosine_df[user_similarity_cosine_df[picked_userid]>=user_similarity_threshold][picked_userid].sort_values(ascending=False)
    # print(f'The similar users for user {picked_userid} are \n', similar_users)

    return similar_users

def get_similar_user_games(picked_userid, similar_users):
    picked_userid_played = matrix[matrix.index == picked_userid].replace(0, float('NaN')).dropna(axis=1)
    picked_userid_played_transposed = picked_userid_played.T

    picked_userid_played_transposed_with_game_title = picked_userid_played_transposed.merge(games, how='left', left_index=True, right_on='app_id')
    picked_userid_played_transposed_with_game_title = picked_userid_played_transposed_with_game_title.iloc[:, :3]

    # print(f'The games the picked user played \n', picked_userid_played_transposed_with_game_title)

    similar_user_games = matrix[matrix.index.isin(similar_users.index)]
    similar_user_games.drop(picked_userid_played.columns,axis=1, inplace=True, errors='ignore')

    similar_user_games_filtered = similar_user_games.loc[:, (similar_user_games != 0).any(axis=0)]

    return similar_user_games_filtered


# # Recommending games

def recommend_games(picked_userid, similar_users, similar_user_games_filtered ):
    item_score = {}

    for i in similar_user_games_filtered.columns:
        if games.loc[games['app_id'] == i, 'rating_encoded'].values[0] > 4:

        # calculated previously 'is_recommend' * 'recommendation_credibility_normalized_log'
            game_rating = similar_user_games_filtered[i]
            total = 0
            count = 0
            for u in similar_users.index:

                if pd.isna(game_rating[u]) == False:

                    # getting the credibility score for the similar user u
                    user_credibility_log = users.loc[users['user_id'] == u, 'user_credibility_log'].values[0]

                    # The more similar the user is to the targeted user, and more credible that user, and higher the game rating, the higher the score
                    score = (similar_users[u] * game_rating[u] * user_credibility_log) * 10000
                    total += score
                    count +=1
            item_score[i] = total / count
    item_score = pd.DataFrame(item_score.items(), columns=['game_id', 'game_score'])

    ranked_item_score = item_score.sort_values(by='game_score', ascending=False)
    ranked_item_score_merged_dataset = ranked_item_score.merge(games, how='left', left_on='game_id', right_on='app_id')

    return ranked_item_score_merged_dataset


# # Evaluation - Kmeans Clustering

def get_user_cluster(picked_userid):
    user_cluster = df_with_clusters[df_with_clusters['user_id'] == picked_userid]['cluster_label'].values

    if len(user_cluster) == 0:
        # print(f"User {picked_userid} is not found in any cluster.")
        pass
    else:
        # print(f"User {picked_userid} belongs to cluster {user_cluster[0]}.")
        pass

    return user_cluster[0]

def get_top_n_popular_games_per_cluster():
    df_with_clusters_with_information = pd.merge(df_with_clusters, games, on='app_id', how='left')
    cluster_game_counts = df_with_clusters_with_information.groupby(['cluster_label', 'app_id', 'positive_ratio_log']).size().reset_index(name='User_Count')

    cluster_games_with_info = pd.merge(cluster_game_counts, games, on='app_id', how='left')

    cluster_game_counts_sorted = cluster_games_with_info.sort_values(by=['cluster_label', 'User_Count'], ascending=[True, False])

    top_n_popular_games_per_cluster = cluster_game_counts_sorted.groupby('cluster_label').head(500)
    # print(top_n_popular_games_per_cluster)

    return top_n_popular_games_per_cluster


def check_against_cluster_list(picked_userid, k, user_cluster, top_n_popular_games_per_cluster, ranked_item_score_merged_dataset):
    user_clusters = [int(x) for x in user_cluster[1:-1].split(',')]

    for cluster in user_clusters:
        user_cluster_info = top_n_popular_games_per_cluster[top_n_popular_games_per_cluster.apply(lambda row: str(cluster) in row['cluster_label'], axis=1)]

        is_in_cluster = ranked_item_score_merged_dataset['game_id'].head(k).isin(user_cluster_info['app_id'])
        true_false_counts = is_in_cluster.value_counts()

        # print(is_in_cluster)
        # print("Accuracy result of checking against cluster: " + str(cluster))
        # print(true_false_counts)

        if len(is_in_cluster) > 0 :
            precision_k_picked_user = sum(is_in_cluster)/len(is_in_cluster)
        else:
            precision_k_picked_user = 0
        # print(precision_k_picked_user)

        return precision_k_picked_user


# # Wrapper

def recommend_and_evaluate_similarity(picked_userid, k=20):
    similar_users = get_similar_users(picked_userid)

    similar_user_games_filtered = get_similar_user_games(picked_userid, similar_users)

    ranked_item_score_merged_dataset = recommend_games(picked_userid, similar_users, similar_user_games_filtered)
    # print(ranked_item_score_merged_dataset[['game_id', 'game_score', 'title']].head(k))

    user_cluster = get_user_cluster(picked_userid)

    top_n_popular_games_per_cluster = get_top_n_popular_games_per_cluster()

    precision_k_picked_user = check_against_cluster_list(picked_userid, k, user_cluster, top_n_popular_games_per_cluster, ranked_item_score_merged_dataset)

    print("Accuracy for user " + str(picked_userid) + ":")
    print(precision_k_picked_user)

    return ranked_item_score_merged_dataset[['game_id', 'game_score']].head(k), precision_k_picked_user


# # Repeat process for N users

# import warnings
# warnings.filterwarnings('ignore')

# # %%
# N = 10
# total = 0
# k= 20

# first_n_user_ids = users.head(N)['user_id'].tolist()
# for user in first_n_user_ids:
#     picked_userid = user
#     precision_k_for_picked_user = recommend_and_evaluate(picked_userid, k)
#     total += precision_k_for_picked_user

# precision_k_n = total/N

# print(f"Precision@K for N users, where K= {k} and N= {N}: ")
# print(precision_k_n)



