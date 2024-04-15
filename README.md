#### BT4222 Final Project Recommendation System Group 9

# Project Title: Steam Game Recommendation System

### Recommendation System Group 9 Members:

Karen Law,
Marco Mah,
Teo Yee Hui,
Tan Zhengkang,
Alistar Tan,
Ryan Koh

## Overview

Welcome to the Steam Game Recommendation System! Our project aims to enhance the game shopping experience on the Steam platform by providing improved game recommendations to users. We utilize various machine learning models to achieve this goal, including similarity-based models, diversity and novelty-based models, and a hybrid model combining both approaches.

## Models

### Similarity-Based Models

Model 1: Content-Based Filtering <br />
Model 2: Collaborative Filtering <br />

- To recommend games using similar games that users played <br />

### Diversity and Novelty-Based Models

Model 3: Collaborative Filtering for Diversity and Novelty <br />
Model 4: Neural Collaborative Filtering for Novelty <br />

### Hybrid Model

Model 5: Hybrid Model <br />

- To provide a combination of games based on the different models

## How to Run

To run our Python code and generate game recommendations, download the entire repository to your local computer, and follow these steps: <br />

### Step 1: Exploratory Data Analysis

Start by running the code to explore the data visually and statistically to gain insights into its characteristics and relationships. <br />

### Step 2: Data Preprocessing and Preparation

### Step 2.1 - Data Preprocessing (game) 
Combined with game metadata file 

### Step 2.2 - Data Preprocessing (user)

Included in the user credibility score 

### Step 2.3 - Data Preprocessing (user_sample_data): 

As the original datasets were too big for our local machines and Google Colab to process, we decided to use a smaller sample of users
for our recommendation system. Hence, we randomly sampled 30000 users from the user dataset. In our respective models, we will filter the recommendations and games data to only contain recommendations and games linked to the 30000 users. 

### Step 2.4 - Data Preprocessing (recommendations) 

Recommendations credibility score

### Step 2.5 - Data Preparation (`clustering.ipynb`): 

This code is to cluster the users into different clusters, to prepare for evaluating the models using precision@k (accuracy).

The results are saved into the folder 'Cleaned Data': 

- Dataset 1: games_cleaned_PCA.csv and games_cleaned.csv 
- Dataset 2: users_with_score.csv 
- Dataset 3: sample_user_data.csv
- Dataset 4: recommendations_with_score.csv
- Dataset 5: clustering.csv 

### Step 3: Models

Step 3.1 - Content-Based Filtering

- To run the model, ensure the following files are in the same directory as the notebook:

   1. recommendations_with_score.csv
   2. sample_user_data.csv
   3. games_cleaned.csv
   4. clustering.csv
   5. item_matrix.py
   6. games_metadata.json

Step 3.2 - Collaborative Filtering (`collaborative_filtering.ipynb`) 

- To run the model, ensure the following files are in the same directory as the notebook:
  1. recommendations_with_score.csv
  2. sample_user_data.csv
  3. games_cleaned.csv
  4. clustering.csv
  5. item_matrix.py
  6. games_metadata.json
    

Step 3.3 - Collaborative Filtering for Diversity and Novelty 

- To run the model, ensure the following files are in the same directory as the notebook:

  1. recommendations_with_score.csv
  2. sample_user_data.csv
  3. games_cleaned.csv
  4. clustering.csv
  5. item_matrix.py
  6. games_metadata.json


Step 3.4 - Neural Collaborative Filtering for Novelty <br />

- To run the model, ensure the following files are in the same directory as the notebook:

  1. recommendations_with_score.csv
  2. sample_user_data.csv
  3. games_cleaned.csv
  4. clustering.csv
  5. item_matrix.py
  6. games_metadata.json

Step 3.5 - Hybrid Model 

- The hybrid model folder contains the hybrid model file with its dependency files. To run the model, execute the code in `hybrid.ipynb`. Ensure the following files are in the same directory as the notebook:

  1. recommendations_with_score.csv
  2. sample_user_data.csv
  3. games_cleaned.csv
  4. clustering.csv
  5. item_matrix.py
  6. games_metadata.json

## Datasets [This part we need to edit to include everything data we need to run the models]

We have the following datasets (from Kaggle) available for use and is stored in 'Original Data from Kaggle' folder: <br />

- Dataset 1: games.csv <br />
- Dataset 2: recommendations.csv <br />
- Dataset 3: users.csv <br />

