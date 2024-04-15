
import pandas as pd
import pickle

# Load datasets
games_df = pd.read_csv("games_cleaned.csv")
df_with_clusters = pd.read_csv("clustering.csv")
#games_pca_df = pd.read_csv("games_cleaned_pca.csv", usecols=['title', 'app_id', 'user_reviews_log', 'positive_ratio_log', 'price_original_log'] + ['PC{}'.format(i) for i in range(1, 116)])
recommendations_df = pd.read_csv("recommendations_with_score.csv", usecols=['user_id','app_id', 'hours_log', 'is_recommended','recommendation_credibility_normalized_log'])
users_df = pd.read_csv("sample_user_data.csv", usecols=['user_id', 'user_credibility_log','reviews'])


games_df = games_df.drop(columns = ["win","mac","linux","date_release","price_final_log","discount"])

# Merging datasets
# Start by merging the recommendations with the game information
merged_df = pd.merge(recommendations_df, users_df, on="user_id")
# # Then merge the user information
final_df = pd.merge(merged_df, games_df, on="app_id")

# # Calculating new scores

import numpy as np

# Here im creating a new variable called "enthusiasm" to represent how many hours the user spent
# in that game relative to the average hours a user spends on that game

# Calculate the average 'hours_log' for each game across all users
average_hours_per_game = final_df.groupby('app_id')['hours_log'].mean().reset_index(name='avg_hours_log')

# Merge this average back into the original dataframe
final_df = final_df.merge(average_hours_per_game, on='app_id')

# Calculate the 'enthusiasm' feature
final_df['enthusiasm'] = final_df['avg_hours_log'] / final_df['hours_log']

final_df['enthusiasm'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Adding a small constant to the denominator
final_df['enthusiasm'] = final_df['avg_hours_log'] / (final_df['hours_log'] + 1e-9)

final_df = final_df.drop(columns = ['avg_hours_log','hours_log'])

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler to scale between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to data and transform it
final_df['user_reviews_log'] = scaler.fit_transform(final_df[['user_reviews_log']])
final_df['positive_ratio_log'] = scaler.fit_transform(final_df[['positive_ratio_log']])
final_df['enthusiasm'] = scaler.fit_transform(final_df[['enthusiasm']])

# A weightage between 0 and 1 for the final score,
# if you want to account for less popular game that are good, put in a smaller number
popularity_weightage = -1

goodness_weightage = .8

# Weightage on enthusiasm, do note that popular multiplayer games tend to have more hours
# hence if you want to debias these games, assign a lower weightage
enthusiasm_weightage = 0.8


final_df['is_recommended'] = final_df['is_recommended'] * ((final_df['user_reviews_log'] * popularity_weightage) + \
                           (final_df['positive_ratio_log'] * goodness_weightage) + \
                           (final_df['enthusiasm'] * enthusiasm_weightage))

# # Creating user-game matrix and calculating cosine similarity

df_recommender = final_df[['user_id', 'app_id', 'is_recommended', 'recommendation_credibility_normalized_log']]

matrix = df_recommender.pivot_table(index='user_id', columns='app_id', values='is_recommended', fill_value=0)

matrix


from sklearn.metrics.pairwise import cosine_similarity

user_dissimilarity_cosine = cosine_similarity(matrix.fillna(0))
user_dissimilarity_cosine_df = pd.DataFrame(user_dissimilarity_cosine, index=matrix.index, columns=matrix.index)
user_dissimilarity_cosine_df



# # Getting similar users and the games they played

def get_dissimilar_users(picked_userid):
    user_dissimilarity_threshold = -0.05
    dissimilar_users = user_dissimilarity_cosine_df[user_dissimilarity_cosine_df[picked_userid]<=user_dissimilarity_threshold][picked_userid].sort_values(ascending=False)
#     print(f'The similar users for user {picked_userid} are \n', dissimilar_users)

    return dissimilar_users

get_dissimilar_users(947)

def get_dissimilar_user_games(picked_userid, dissimilar_users):
    picked_userid_played = matrix[matrix.index == picked_userid].replace(0, float('NaN')).dropna(axis=1)
    picked_userid_played_transposed = picked_userid_played.T

    picked_userid_played_transposed_with_game_title = picked_userid_played_transposed.merge(games_df, how='left', left_index=True, right_on='app_id')
    picked_userid_played_transposed_with_game_title = picked_userid_played_transposed_with_game_title.iloc[:, :3]

#     print(f'The games the picked user played \n', picked_userid_played_transposed_with_game_title)

    dissimilar_user_games = matrix[matrix.index.isin(dissimilar_users.index)]
    dissimilar_user_games.drop(picked_userid_played.columns,axis=1, inplace=True, errors='ignore')

    dissimilar_user_games_filtered = dissimilar_user_games.loc[:, (dissimilar_user_games != 0).any(axis=0)]

    return dissimilar_user_games_filtered

# # Recommending Games

def recommend_games(picked_userid, dissimilar_users, dissimilar_user_games_filtered):
    item_score = {}

    for i in dissimilar_user_games_filtered.columns:
        if games_df.loc[games_df['app_id'] == i, 'rating_encoded'].values[0] > 4:
#             print(i)
        # calculated previously 'is_recommend' * 'recommendation_credibility_normalized_log'
            game_rating = dissimilar_user_games_filtered[i]
            total = 0
            count = 0
            for u in dissimilar_users.index:

                if pd.isna(game_rating[u]) == False:

                    # getting the credibility score for the dissimilar user u
                    user_credibility_log = users_df.loc[users_df['user_id'] == u, 'user_credibility_log'].values[0]

                    # The more dissimilar the user is to the targeted user, and more credible that user, and higher the game rating, the higher the score
                    score = (-1* dissimilar_users[u] * game_rating[u] * user_credibility_log) * 10000
                    total += score
                    count +=1
            item_score[i] = total / count
    item_score = pd.DataFrame(item_score.items(), columns=['game_id', 'game_score'])

    ranked_item_score = item_score.sort_values(by='game_score', ascending=False)
    ranked_item_score_merged_dataset = ranked_item_score.merge(games_df, how='left', left_on='game_id', right_on='app_id')


    return ranked_item_score_merged_dataset


#picked_userid = 947
#dissimilar_users = get_dissimilar_users(picked_userid)

#dissimilar_user_games_filtered = get_dissimilar_user_games(picked_userid, dissimilar_users)

#ranked_item_score_merged_dataset = recommend_games(picked_userid, dissimilar_users, dissimilar_user_games_filtered)
#ranked_item_score_merged_dataset

# # Evaluation - kmeans clustering

def get_user_cluster(picked_userid):
    user_cluster = df_with_clusters[df_with_clusters['user_id'] == picked_userid]['cluster_label'].values

    if len(user_cluster) == 0:
#         print(f"User {picked_userid} is not found in any cluster.")
        pass
    else:
#         print(f"User {picked_userid} belongs to cluster {user_cluster[0]}.")
        pass

    return user_cluster[0]

def get_top_n_popular_games_per_cluster():
    df_with_clusters_with_information = pd.merge(df_with_clusters, games_df, on='app_id', how='left')
    cluster_game_counts = df_with_clusters_with_information.groupby(['cluster_label', 'app_id', 'positive_ratio_log']).size().reset_index(name='User_Count')

    cluster_games_with_info = pd.merge(cluster_game_counts, games_df, on='app_id', how='left')

    cluster_game_counts_sorted = cluster_games_with_info.sort_values(by=['cluster_label', 'User_Count'], ascending=[True, False])

    top_n_popular_games_per_cluster = cluster_game_counts_sorted.groupby('cluster_label').head(500)
#     print(top_n_popular_games_per_cluster)

    return top_n_popular_games_per_cluster

def check_against_cluster_list(picked_userid, k, user_cluster, top_n_popular_games_per_cluster, ranked_item_score_merged_dataset):
    user_clusters = [int(x) for x in user_cluster[1:-1].split(',')]

    for cluster in user_clusters:
        user_cluster_info = top_n_popular_games_per_cluster[top_n_popular_games_per_cluster.apply(lambda row: str(cluster) in row['cluster_label'], axis=1)]

        is_in_cluster = ranked_item_score_merged_dataset['game_id'].head(k).isin(user_cluster_info['app_id'])
        true_false_counts = is_in_cluster.value_counts()

#         print(is_in_cluster)
#         print("Accuracy result of checking against cluster: " + str(cluster))
#         print(true_false_counts)

        if len(is_in_cluster) > 0 :
            precision_k_picked_user = sum(is_in_cluster)/len(is_in_cluster)
        else:
            precision_k_picked_user = 0
#         print(precision_k_picked_user)

        return precision_k_picked_user



# # Wrapper

def recommend_and_evaluate_diversity(picked_userid, k =20):
    dissimilar_users = get_dissimilar_users(picked_userid)

    dissimilar_user_games_filtered = get_dissimilar_user_games(picked_userid, dissimilar_users)

    ranked_item_score_merged_dataset = recommend_games(picked_userid, dissimilar_users, dissimilar_user_games_filtered)
    print(ranked_item_score_merged_dataset[['game_id', 'game_score']].head(k))

    user_cluster = get_user_cluster(picked_userid)

    top_n_popular_games_per_cluster = get_top_n_popular_games_per_cluster()

    precision_k_picked_user = check_against_cluster_list(picked_userid, k, user_cluster, top_n_popular_games_per_cluster, ranked_item_score_merged_dataset)

    print("Accuracy for user " + str(picked_userid) + ":")
    print(precision_k_picked_user)

    return ranked_item_score_merged_dataset[['game_id', 'game_score']].head(k), precision_k_picked_user


#picked_userid = 947
#dissimilar_users = get_dissimilar_users(picked_userid)

#dissimilar_user_games_filtered = get_dissimilar_user_games(picked_userid, dissimilar_users)

#ranked_item_score_merged_dataset = recommend_games(picked_userid, dissimilar_users, dissimilar_user_games_filtered)

#ranked_item_score_merged_dataset


# ## ranked_item_score_merged_dataset produces the list below

#print(recommend_and_evaluate_diversity(947))

## Repeat process for N users


# import warnings
# warnings.filterwarnings('ignore')

# N = 10
# total = 0
# k= 20

# first_n_user_ids = users_df.head(N)['user_id'].tolist()
# for user in first_n_user_ids:
#     picked_userid = user
#     precision_k_for_picked_user = recommend_and_evaluate_diversity(picked_userid, k)
#     total += precision_k_for_picked_user

# precision_k_n = total/N

# print(f"Precision@K for N users, where K= {k} and N= {N}: ")
# print(precision_k_n)

