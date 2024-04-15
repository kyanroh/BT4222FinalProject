# -*- coding: utf-8 -*-
"""neural_collaborative_filtering_with_score.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z0a_yHx-ihdh4WCNsyjabF9kBK8RVpGH

# Step 1 Data loading
"""

import pandas as pd

# Load datasets
games_df = pd.read_csv("games_cleaned.csv")
#games_pca_df = pd.read_csv("games_cleaned_PCA.csv", usecols=['title', 'app_id', 'user_reviews_log', 'positive_ratio_log', 'price_original_log'] + ['PC{}'.format(i) for i in range(1, 116)])
recommendations_df = pd.read_csv("recommendations_with_score.csv", usecols=['user_id','app_id', 'hours_log', 'is_recommended','recommendation_credibility_normalized_log'])
users_df = pd.read_csv("sample_user_data.csv", usecols=['user_id', 'user_credibility_log','reviews'])
df_with_clusters = pd.read_csv("clustering.csv")

games_df = games_df.drop(columns = ["win","mac","linux","date_release","price_final_log","discount"])

# Merging datasets
# Start by merging the recommendations with the game information
merged_df = pd.merge(recommendations_df, users_df, on="user_id")
# # Then merge the user information
final_df = pd.merge(merged_df, games_df, on="app_id")

final_df = final_df[final_df['rating_encoded'] >= 4]

"""# Step 2: Calculating new features"""

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

"""# Step 3: Scaling the different scores"""

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler to scale between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to data and transform it
final_df['user_reviews_log'] = scaler.fit_transform(final_df[['user_reviews_log']])
final_df['positive_ratio_log'] = scaler.fit_transform(final_df[['positive_ratio_log']])
final_df['enthusiasm'] = scaler.fit_transform(final_df[['enthusiasm']])

"""# Step 4: Calculating a new score for target"""

# A weightage between 0 and 1 for the final score,
# if you want to account for games with less reviews, put in a smaller number
popularity_weightage = .3

goodness_weightage = 1

# Weightage on enthusiasm, do note that popular multiplayer games tend to have more hours
enthusiasm_weightage = 0.3


final_df['is_recommend'] = final_df['is_recommended'] * \
                           ((final_df['user_reviews_log'] * popularity_weightage) + \
                           (final_df['positive_ratio_log'] * goodness_weightage) + \
                           (final_df['enthusiasm'] * enthusiasm_weightage))


final_df = final_df.drop(columns = ['positive_ratio_log','user_reviews_log','enthusiasm'])

"""# Step 5: Encoding the user id and app id for model"""

# Since the model cannot take random user_id and app_id, i need to encode them into a
# list from 0 to number of users

import pandas as pd

# Encode 'user_id' since the model cannot take random user_ids
user_id_encoder = {id: idx for idx, id in enumerate(final_df['user_id'].unique())}
final_df['user_id_encoded'] = final_df['user_id'].map(user_id_encoder)

# Encode 'app_id' since the model cannot take random app_ids
app_id_encoder = {id: idx for idx, id in enumerate(final_df['app_id'].unique())}
final_df['app_id_encoded'] = final_df['app_id'].map(app_id_encoder)

# Decoder 'user_id' for later usage when the functions receive a list of original id
user_id_decoder = {v: k for k, v in user_id_encoder.items()}

# Here im scaling the target of the model to avoid any extraordinarily big or small numbers

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler to scale between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to data and transform it
final_df['is_recommend_normalized'] = scaler.fit_transform(final_df[['is_recommend']])

# Check the first few rows to see the normalization
print(final_df[['is_recommend', 'is_recommend_normalized']].head())

"""# Step 6: Building the NCF Model"""

# A NCF model is used here to try to model the non-linear interactions between the user and games
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Parameters for embedding layers
num_users = final_df['user_id_encoded'].nunique()
num_items = final_df['app_id_encoded'].nunique()
embedding_size = 10

# User and Item Inputs
user_id_input = Input(shape=(1,), name='user_id_input')
item_id_input = Input(shape=(1,), name='item_id_input')

# Embeddings
user_embedding_gmf = Embedding(num_users, embedding_size, embeddings_regularizer=l2(1e-6), name='user_embedding_gmf')(user_id_input)
item_embedding_gmf = Embedding(num_items, embedding_size, name='item_embedding_gmf')(item_id_input)

user_embedding_mlp = Embedding(num_users, embedding_size * 2, name='user_embedding_mlp')(user_id_input)
item_embedding_mlp = Embedding(num_items, embedding_size * 2, name='item_embedding_mlp')(item_id_input)

# Flatten embeddings
user_vector_gmf = Flatten(name='flatten_user_gmf')(user_embedding_gmf)
item_vector_gmf = Flatten(name='flatten_item_gmf')(item_embedding_gmf)

user_vector_mlp = Flatten(name='flatten_user_mlp')(user_embedding_mlp)
item_vector_mlp = Flatten(name='flatten_item_mlp')(item_embedding_mlp)

# GMF part (simple element-wise multiplication)
gmf_vector = Dot(axes=1, normalize=False, name='gmf_dot')([user_vector_gmf, item_vector_gmf])

# MLP part (concatenation followed by dense layers)
mlp_vector = Concatenate(name='concatenate_mlp')([user_vector_mlp, item_vector_mlp])
mlp_vector = Dense(32, activation='relu', kernel_regularizer=l2(1e-6), name='dense_layer_1')(mlp_vector)  # Reduced complexity

# Concatenate GMF and MLP parts remain unchanged
combined_vector = Concatenate(name='concatenate_gmf_mlp')([gmf_vector, mlp_vector])
predictions = Dense(1, activation=None, name='output_layer')(combined_vector)

# Define the model
model = Model(inputs=[user_id_input, item_id_input], outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_squared_logarithmic_error'])


# Model summary
model.summary()

"""# Step 7: Training the model"""

from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
X = final_df[['user_id_encoded', 'app_id_encoded']]
y = final_df['is_recommend_normalized']

# Split the data - adjust the test size as needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit([X_train.user_id_encoded, X_train.app_id_encoded], y_train,
          batch_size=128, epochs=10,
          validation_split=0.1)

# Evaluate the model
model.evaluate([X_test.user_id_encoded, X_test.app_id_encoded], y_test)

"""# Step 8: Creating sparse matrix for all user-app pair"""

# Here a COO-format sparse matrix is created for all user-app pair because a dense matrix
# is way too big

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

data = np.ones(len(final_df), dtype=int)

# Create a COO-format sparse matrix from user-item interactions
sparse_matrix = coo_matrix((data, (final_df['user_id_encoded'], final_df['app_id_encoded'])))

sparse_matrix_csr = sparse_matrix.tocsr()

# Here a function is defined to get all the unteracted user-game pair to make predictions
def get_uninteracted_items(user_id, sparse_matrix, n_items):
    """Get a list of item IDs that the given user has not interacted with."""
    interacted_items = sparse_matrix[user_id].nonzero()[1]  # Get the indices of interacted items for the user
    all_items = np.arange(n_items)  # Array of all item IDs
    uninteracted_items = np.setdiff1d(all_items, interacted_items)  # Find items that the user hasn't interacted with
    return uninteracted_items

# Example usage
n_users = final_df['user_id_encoded'].max() + 1  # Assuming IDs start from 0
n_items = final_df['app_id_encoded'].max() + 1  # Assuming IDs start from 0

user_id = 2  # Example user ID
uninteracted_items = get_uninteracted_items(user_id, sparse_matrix_csr, n_items)
print(f"User {user_id} has not interacted with {len(uninteracted_items)} items.")

"""# Step 9: Generating Recommendation"""

# Here is a function that generates recommendations given a user_id, note that
# this user_id is the original user id, which will be encoded in the following
# chunk for the model to work.

def generate_recommendations_for_user(user_id, sparse_matrix, model, n_items, top_n=20):

    # Step 0 : Encode the user id
    user_id = user_id_encoder[user_id]

    # Step 1: Identify uninteracted items
    uninteracted_items = get_uninteracted_items(user_id, sparse_matrix, n_items)

    # Prepare user and item IDs for prediction
    user_ids = np.array([user_id] * len(uninteracted_items))
    item_ids = np.array(uninteracted_items)

    # Predict preferences for the uninteracted items
    predictions = model.predict([user_ids, item_ids])


    # Common fix for an extra dimension in predictions
    predictions = predictions.squeeze()

    # Proceed with combining arrays and the rest of the function...
    combined = np.rec.fromarrays([item_ids, predictions], names='item_id,prediction')

    # Step 3: Sort items by predicted preference
    sorted_items = np.sort(combined, order='prediction')[::-1]

    # Step 4: Filter to ensure unique item recommendations
    # This step is crucial to remove duplicates while keeping the highest ranked recommendation for each item.
    _, unique_indices = np.unique(sorted_items['item_id'], return_index=True)

    # Select top unique recommendations based on the unique_indices, up to top_n
    unique_top_indices = np.sort(unique_indices)[:top_n]
    top_item_ids = sorted_items[['item_id','prediction']][unique_top_indices]


    return top_item_ids

"""# Step 10: Creating Dictionaries to map back the encoded user and app ids"""

# Here a function is defined to map the encoded app id into original app id, vice versa.

app_id_to_title = final_df.set_index('app_id_encoded')['title'].to_dict()
app_id_to_real_app_id = final_df.set_index('app_id_encoded')['app_id'].to_dict()

"""## Testing Single recommendation"""

chosen_id = 947

recommended_items = pd.DataFrame(generate_recommendations_for_user(chosen_id, sparse_matrix_csr, model, n_items, top_n = 20))

recommended_items['item_id'] = recommended_items['item_id'].replace(app_id_to_real_app_id)

print(f"These are the game recommendations for user {chosen_id}.")
print(recommended_items)

"""# Step 11: Evaluation- Kmeans Clustering

"""

# Here a function is defined to the cluster games of the chosen user
def get_user_cluster(picked_userid):
    user_cluster = df_with_clusters[df_with_clusters['user_id'] == picked_userid]['cluster_label'].values

    if len(user_cluster) == 0:
        # print(f"User {picked_userid} is not found in any cluster.")
        pass
    else:
        # print(f"User {picked_userid} belongs to cluster {user_cluster[0]}.")
        pass

    return user_cluster[0]

# Here is a function to retrieve the top games in the user cluster
def get_top_n_popular_games_per_cluster():
    df_with_clusters_with_information = pd.merge(df_with_clusters, games_df, on='app_id', how='left')
    cluster_game_counts = df_with_clusters_with_information.groupby(['cluster_label', 'app_id', 'positive_ratio_log']).size().reset_index(name='User_Count')

    cluster_games_with_info = pd.merge(cluster_game_counts, games_df, on='app_id', how='left')

    cluster_game_counts_sorted = cluster_games_with_info.sort_values(by=['cluster_label', 'User_Count'], ascending=[True, False])

    top_n_popular_games_per_cluster = cluster_game_counts_sorted.groupby('cluster_label').head(500)
    # print(top_n_popular_games_per_cluster)

    return top_n_popular_games_per_cluster

# Here is a function that checks the recommended list of the NCF vs the top cluster games
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

"""# Step 12: Wrapper"""

def recommend_and_evaluate_neural(picked_userid, k= 20):

    recommended_items = pd.DataFrame(generate_recommendations_for_user(picked_userid, sparse_matrix_csr, model, n_items))

    recommended_items['item_id'] = recommended_items['item_id'].replace(app_id_to_real_app_id)

    # Rename the 'item_id' column to 'game_id'
    recommended_items.rename(columns={'item_id': 'game_id'}, inplace=True)

    recommended_items.rename(columns={'prediction': 'game_score'}, inplace=True)

    ranked_item_score = recommended_items.sort_values(by='game_score', ascending=False)
    ranked_item_score_merged_dataset = ranked_item_score.merge(games_df, how='left', left_on='game_id', right_on='app_id')

    print(ranked_item_score_merged_dataset[['game_id', 'game_score', 'title']].head(k))

    user_cluster = get_user_cluster(picked_userid)

    top_n_popular_games_per_cluster = get_top_n_popular_games_per_cluster()

    precision_k_picked_user = check_against_cluster_list(picked_userid, k, user_cluster, top_n_popular_games_per_cluster, ranked_item_score_merged_dataset)

    print("Accuracy for user " + str(picked_userid) + ":")
    print(precision_k_picked_user)

    return ranked_item_score_merged_dataset[['game_id', 'game_score']].head(k), precision_k_picked_user

'''
N = 30
total = 0
k= 20

first_n_user_ids = users_df.head(N)['user_id'].tolist()
for user in first_n_user_ids:
    picked_userid = user
    try:
        precision_k_for_picked_user = recommend_and_evaluate_neural(947)(picked_userid, k)
        total += precision_k_for_picked_user
    except Exception as e:
        continue

precision_k_n = total/N

print(f"Precision@K for N users, where K= {k} and N= {N}: ")
print(precision_k_n)

"""# Calculate Novelty Score"""

# Read games data
games_df = pd.read_csv('games_cleaned.csv')

# Create the popularity_score dictionary
popularity_score = {}

# Populate the popularity_score dictionary based on the sample data
for index, row in games_df.iterrows():
    popularity_score[row['app_id']] = row['rating_encoded']/8

# Print the popularity_score dictionary
print(popularity_score)

pop_dict = popularity_score

def novelty_metric(rec_list, pop_dict):
    pop_sum = []  # List to store popularity scores of recommended items
    for item in rec_list:
        if item in pop_dict.keys():  # Check if the item exists in the popularity dictionary
            pop_sum.append(pop_dict[item])  # Add the popularity score of the item to the list
    return np.mean(pop_sum)  # Calculate and return the mean popularity score of recommended items

# Extract the list of recommended app_IDs from the recommended_items DataFrame
rec_list = recommended_items['item_id'].to_list()

# Calculate the novelty score for the recommended items using the novelty_metric function
novelty_metric(rec_list, pop_dict)

# Set the number of recommendations and the number of users
n_recommendations = 20
N = 30

# Initialize total novelty score
total_novelty_score = 0

# Get the first N user IDs
first_n_user_ids = users_df.head(N)['user_id'].tolist()

# Iterate over each user
for user in first_n_user_ids:
    # Generate recommended item IDs for the user
    recommended_items = pd.DataFrame(generate_recommendations_for_user(user, sparse_matrix_csr, model, n_items, top_n=n_recommendations))

    # Map app IDs to real app IDs
    recommended_items['item_id'] = recommended_items['item_id'].replace(app_id_to_real_app_id)

    # Extract the list of recommended item IDs
    rec_list = recommended_items['item_id'].to_list()

    # Check if rec_list is empty
    if not rec_list:
        novelty_score = 0  # If there are no recommendations, novelty score is 0
    else:
        # Calculate novelty score for the recommendations
        novelty_score = novelty_metric(rec_list, pop_dict)

    # Print the user ID and its novelty score
    print(user, novelty_score)

    # Accumulate the novelty score
    total_novelty_score += novelty_score

# Calculate the average novelty score
average_novelty_score = total_novelty_score / N

# Print the average novelty score
print("Average novelty score for {N} users is: ", average_novelty_score)

def get_novelty_score(user_id):
    # Generate recommended item IDs for the user
    recommended_items = pd.DataFrame(generate_recommendations_for_user(user_id, sparse_matrix_csr, model, n_items, top_n=n_recommendations))

    # Map app IDs to real app IDs
    recommended_items['item_id'] = recommended_items['item_id'].replace(app_id_to_real_app_id)

    # Extract the list of recommended item IDs
    rec_list = recommended_items['item_id'].to_list()

    # Calculate the novelty score for the recommended items
    novelty_score = novelty_metric(rec_list, pop_dict)

    return novelty_score

"""Calculate Diversity Score"""

# Import the function get_item_matrix from the item_matrix module
from item_matrix import get_item_matrix

# Call the get_item_matrix function to obtain the item similarity matrix
item_sim_matrix = get_item_matrix()

def ils_metric(rec_list, item_sim_matrix):
    sim_temp = 0  # Initialize a temporary variable to store the similarity sum
    for i in range(0, len(rec_list)):
        for j in range(i + 1, len(rec_list)):
            # Check if item j is in the similarity matrix for item i
            if rec_list[j] in item_sim_matrix[rec_list[i]]:
                # If yes, add the similarity score to sim_temp
                sim_temp += item_sim_matrix[rec_list[i]][rec_list[j]]
    # Calculate the ILS score by subtracting the normalized similarity sum from 1
    return 1 - (sim_temp / (len(rec_list) * (len(rec_list) - 1)))

# Set the number of recommendations and the number of users
n_recommendations = 20
N = 30

# Initialize total diversity score
total_diversity_score = 0

# Get the first N user IDs
first_n_user_ids = users_df.head(N)['user_id'].tolist()

# Iterate over each user
for user in first_n_user_ids:
    # Generate recommended item IDs for the user
    recommended_items = pd.DataFrame(generate_recommendations_for_user(user, sparse_matrix_csr, model, n_items, top_n=n_recommendations))

    # Map app IDs to real app IDs
    recommended_items['item_id'] = recommended_items['item_id'].replace(app_id_to_real_app_id)

    # Extract the list of recommended item IDs
    rec_list = recommended_items['item_id'].to_list()

    # Check if rec_list is empty
    if not rec_list:
        diversity_score = 0  # If there are no recommendations, diversity score is 0
    else:
        # Calculate diversity score for the recommendations using the ILS metric
        diversity_score = ils_metric(rec_list, item_sim_matrix)

    # Print user ID and diversity score
    print(user, diversity_score)

    # Accumulate diversity score
    total_diversity_score += diversity_score

# Calculate average diversity score
average_diversity_score = total_diversity_score / N

# Print the average diversity score
print("The average diversity score for {N} users is: ", average_diversity_score)
'''