#### BT4222 Final Project Recommendation System Group 9
# Project Title: Steam Game Recommendation System
### Members: 
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
### Diversity and Novelty-Based Models
Model 3: Collaborative Filtering Novelty <br />
Model 4: Neural Collaborative Filtering <br />
### Hybrid Model
Model 5: Hybrid Model <br />
## How to Run
To run our Python code and generate game recommendations, download the entire repository to your local computer, and follow these steps: <br />
### Step 1: Extrapolatory Data Analysis 
Start by running the code to explore the data visually and statistically to gain insights into its characteristics and relationships. <br />
### Step 2: Data Preprocessing and Preparation
In this step, we have 3 different codes to clean up 3 different datasets which are:<br />
Step 2.1 - Data Preprocessing (user_sample???): Get 30,000 users?? <br />
Step 2.2 - Data Preprocessing (game): <br />
Step 2.3 - Data Preprocessing (user): <br />
Step 2.4 - Data Preprocessing (recommendation): In this code, feature engineering and feature crosses ????Details of the steps were commented at the top. <br />
Step 2.5 - Data Preparation (`clustering.ipynb`): This code is to cluster the users into different clusters, to prepare for evaluating the models for precision@k (accuracy). 

The results are saved into the folder 'Cleaned Data': <br />

Dataset 1: <br />
Dataset 2: <br />
Dataset 3: <br />
Dataset 4: <br />
Dataset 5: clustering.csv

### Step 3: Models
Step 3.1 - Content-Based Filtering <br />
Step 3.2 - Collaborative Filtering (`collaborative_filtering.ipynb`) <br />
- To run the model, ensure the following files are in the same directory with the notebook:
  1. recommendations_with_score.csv
  2. sample_user_data.csv
  3. games_claened.csv
  4. clustering.csv
  5. item_matrix.py
<br/>
Step 3.3 - Collaborative Filtering Novelty <br />
Step 3.4 - Neural Collaborative Filtering <br />
Step 3.5 - Hybrid Model <br />



## Datasets [This part we need to edit to include everything data we need to run the models]
We have the following datasets available for use: <br />

Dataset 1: games.csv <br />
Dataset 2: recommendations.csv <br />
Dataset 3: users.csv <br />

## Contribution


## Running the code
