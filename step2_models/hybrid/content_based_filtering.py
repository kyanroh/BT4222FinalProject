
#Importing Package
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#from google.colab import drive
#drive.mount('/content/drive')

#games_metadata = pd.read_json("/content/drive/MyDrive/BT4222/Data/games_metadata.json", lines=True)
#games_data = pd.read_csv("/content/drive/MyDrive/BT4222/Data/games.csv")
games_metadata = pd.read_json("games_metadata.json", lines=True)
#games_data = pd.read_csv("games_cleaned_PCA.csv")
recommendations_data = pd.read_csv("recommendations_with_score.csv")
users_data = pd.read_csv("sample_user_data.csv")
cluster_data = pd.read_csv("clustering.csv")

#Game Metadata
games_metadata.head(5)

#Game Dataset
games_data.head(5)

#Truncate Game Dataset
games_data = games_data[['app_id','title', 'rating_encoded', 'positive_ratio_log']]
games_data.head(5)

print(len(users_data))
users_data.head(35)

#Recommendations Dataset
recommendations_data.head(5)

#Filter Recommendations only with users in the user dataset
recommendations_data = recommendations_data[recommendations_data['user_id'].isin(users_data['user_id'])]
recommendations_data = recommendations_data.sort_values(by='user_id', ascending=True)
recommendations_data = recommendations_data.reset_index(drop=True)
print(len(recommendations_data))
recommendations_data.head(5)

#Merge Game Data and Game Metadata
games_df = pd.merge(games_data, games_metadata, how='inner', on='app_id')
print(len(games_df))

#Remove games that have no descriptions or tags
games_df = games_df[~((games_df['description'] == "") | (games_df['tags'].apply(len) == 0))]
games_df = games_df.sort_values(by=['app_id'])
games_df = games_df.reset_index(drop=True)
print(len(games_df))
games_df.head()

#Truncating recommendation data
recommendations_short_data = recommendations_data[['app_id','user_id','is_recommended']]
recommendations_short_data =  recommendations_short_data[recommendations_short_data['app_id'].isin(games_df['app_id'])]
recommendations_short_data = recommendations_short_data.sort_values(by='user_id', ascending=True)
recommendations_short_data = recommendations_short_data.reset_index(drop=True)
recommendations_short_data.head(5)

cluster_data = cluster_data.sort_values(by='user_id', ascending=True)
cluster_data =  cluster_data[cluster_data['app_id'].isin(games_df['app_id'])]
cluster_data.head(10)

#Number of games recommended by each user
recommended_df = recommendations_short_data[recommendations_short_data['is_recommended'] == 1]

# Group by 'user_id' and calculate the sum of 'is_recommended' (True) for each user
user_recommended_counts = recommended_df.groupby('user_id').size().reset_index(name='recommendation_count')

user_recommended_counts

#Feed features for vectorizer, using title, description, and tags of games
def clean_data_list(x):
    return str(' '.join(x))

games_df['tags'] = games_df['tags'].apply(clean_data_list)
games_df['combined'] = games_df['description'] + ' ' + games_df['tags']
games_df.head(5)



# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(games_df['combined'])

# Calculate Cosine Similarity
def cosine_similarity_n_space(m1, m2, batch_size=100):
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break # cause I'm too lazy to elegantly handle edge cases
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        ret[start: end] = sim
    return ret

cosine_sim = cosine_similarity_n_space(tfidf_matrix, tfidf_matrix)

# Build the recommendation system

#If no liked games
def recommend_games_1(game_id, cosine_sim, df):
    # Get the index of the game
    game_index = df[df['app_id'] == game_id].index[0]
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[game_index]))
    # Sort games based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top N similar games
    top_similar_games = sim_scores[1:11]  # Exclude the first item which is the game itself
    # Get recommended game titles and similarity scores
    recommended_games = [{'title': df.iloc[sim_game[0]]['title'], 'cosine_sim': sim_game[1]} for sim_game in top_similar_games]
    # Create a DataFrame for recommended games
    recommended_df = pd.DataFrame(recommended_games)
    recommended_df = recommended_df.sort_values(by='cosine_sim', ascending=False)
    recommended_df.reset_index(drop=True, inplace=True)
    recommended_df.index = recommended_df.index + 1
    return recommended_df

#If have at least one liked game
def recommend_games_2(user_id, cosine_sim, df, top_n):
    #Filter games out with 4.0 or less rating
    filtered_df = recommendations_short_data[(recommendations_short_data['user_id'] == user_id)
                                             & (recommendations_short_data['is_recommended'] == True)]
    app_ids = filtered_df['app_id'].tolist()
    #print(app_ids)
    recommended_games_list = []
    for app_id in app_ids:
        # Get the index of the game
        game_index = df[df['app_id'] == app_id].index[0]
        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[game_index]))
        # Sort games based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get top N similar games
        top_similar_games = sim_scores[1:top_n+1]  # Exclude the first item which is the game itself
        # Get recommended game titles and similarity scores
        recommended_games = [{'title': df.iloc[sim_game[0]]['title'], 'cosine_sim': sim_game[1]} for sim_game in top_similar_games]
        # Append recommended games to list
        recommended_games_list.extend(recommended_games)

    recommended_df = pd.DataFrame(recommended_games_list)
    recommended_df = recommended_df.sort_values(by='cosine_sim', ascending=False)
    recommended_df = pd.merge(recommended_df, games_df[['title', 'app_id']], on='title', how='left')
    recommended_df = recommended_df[['app_id', 'title', 'cosine_sim']]
    recommended_df.reset_index(drop=True, inplace=True)
    recommended_df.index = recommended_df.index + 1
    return recommended_df.head(top_n)

# Example usage of the recommendation system
result = 'No liked games'

def recommend(user_id, top_n):
    if user_id in user_recommended_counts['user_id'].values:
        filtered_df = recommendations_short_data[(recommendations_short_data['user_id'] == user_id) & (recommendations_short_data['is_recommended'] == True)]
        app_ids = filtered_df['app_id'].tolist()
        result = games_df[games_df['app_id'].isin(app_ids)]
        result = result[result['rating_encoded'] > 4.0]
        output = recommend_games_2(user_id, cosine_sim, games_df, top_n)
    else:
        game_id = games_df['app_id'].sample().values[0]
        filtered_games_df = games_df[games_df['rating_encoded'] > 4.0]
        output = recommend_games_1(game_id, cosine_sim, filtered_games_df)

    #print(result[['app_id', 'title']])
    return output


#recommended_games_df = recommend_games_1(game_title, cosine_sim, games_df)
#recommended_games_df.reset_index(drop=True, inplace=True)
#recommended_games_df.index = recommended_games_df.index + 1
#print(recommended_games_df)

recommend(14305218, 10)

cosine_sim



"""result

Evaluation
"""

games_list = games_df['app_id'].tolist()
user_similarity_cosine_df = pd.DataFrame(cosine_sim)
user_similarity_cosine_df.index = games_list
user_similarity_cosine_df.columns = games_list


user_similarity_cosine_df

def get_user_cluster(picked_userid):
    user_cluster = cluster_data[cluster_data['user_id'] == picked_userid]['cluster_label'].values

    if len(user_cluster) == 0:
        # print(f"User {picked_userid} is not found in any cluster.")
        pass
    else:
        # print(f"User {picked_userid} belongs to cluster {user_cluster[0]}.")
        pass

    return user_cluster[0]

def get_top_n_popular_games_per_cluster():
    df_with_clusters_with_information = pd.merge(cluster_data, games_df, on='app_id', how='left')
    cluster_game_counts = df_with_clusters_with_information.groupby(['cluster_label', 'app_id', 'positive_ratio_log']).size().reset_index(name='User_Count')

    cluster_games_with_info = pd.merge(cluster_game_counts, games_df, on='app_id', how='left')

    cluster_game_counts_sorted = cluster_games_with_info.sort_values(by=['cluster_label', 'User_Count'], ascending=[True, False])
    top_n_popular_games_per_cluster = cluster_game_counts_sorted.groupby('cluster_label').head(100000)

    return top_n_popular_games_per_cluster

g = get_top_n_popular_games_per_cluster()
print(len(g['app_id'].unique()))
print(len(g))
print(g['cluster_label'].value_counts())

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

"""Wrapper"""

def recommend_and_evaluate_content(picked_userid, k=20):

    ranked_item_score_merged_dataset = recommend(picked_userid, k)
    ranked_item_score_merged_dataset.rename(columns={'app_id': 'game_id', 'cosine_sim': 'game_score'}, inplace=True)
    # print(ranked_item_score_merged_dataset[['game_id', 'game_score', 'title']].head(k))

    user_cluster = get_user_cluster(picked_userid)

    top_n_popular_games_per_cluster = get_top_n_popular_games_per_cluster()

    precision_k_picked_user = check_against_cluster_list(picked_userid, k, user_cluster, top_n_popular_games_per_cluster, ranked_item_score_merged_dataset)

    print("Accuracy for user " + str(picked_userid) + ":")
    print(precision_k_picked_user)

    return ranked_item_score_merged_dataset[['game_id', 'game_score']].head(k), precision_k_picked_user

# recommend_and_evaluate(947)

'''
"""Repeat process for N users"""

import warnings
warnings.filterwarnings('ignore')

N = 30
total = 0
k= 20

first_n_user_ids = users_data.head(N)['user_id'].tolist()
for user in first_n_user_ids:
    picked_userid = user
    try:
        precision_k_for_picked_user = recommend_and_evaluate(picked_userid, k)
        total += precision_k_for_picked_user
    except Exception as e:
        continue

precision_k_n = total/N

print(f"Precision@K for N users, where K= {k} and N= {N}: ")
print(precision_k_n)
'''