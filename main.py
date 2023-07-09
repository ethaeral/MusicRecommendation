#############################################
# User_User Optimized Recommendation        #
# Popular Song Recommendation on Cold Start #
#############################################

'''
' Set Up
' Importing Libs and Datasets 
'''

from surprise.prediction_algorithms.knns import KNNBasic
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise import accuracy
import surprise
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import sklearn
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
count_df = pd.read_csv('./count_data.csv')
song_df = pd.read_csv('./song_data.csv')

'''
' Data Cleaning 
' Consolidate data, drop multiples, null values, and insignificant entries
'''

# Merge count_df and song_df data on "song_id". Drop duplicates from song_df data simultaneously
df = pd.merge(count_df, song_df, on='song_id', how='left').drop_duplicates(
    subset=['user_id', 'song_id'])

# Drop the column 'Unnamed: 0'
df = df.drop(['Unnamed: 0'], axis=1)

# Label Encode Merged Data
le = LabelEncoder()
df['user_id'] = le.fit_transform(df['user_id'])
df['song_id'] = le.fit_transform(df['song_id'])


def ratings_count(df, attr):
    # Get the column containing the users
    entries = df[attr]

    # Create a dictionary from users to their number of songs
    ratings_count = dict()

    for entry in entries:
        # If we already have the user, just add 1 to their rating count
        if entry in ratings_count:
            ratings_count[entry] += 1
        # Otherwise, set their rating count to 1
        else:
            ratings_count[entry] = 1
    return ratings_count


def remove_entries(df, cutoff, attr, ratings_count):
    # Create a list of users who need to be removed
    remove_entries = []

    for entry, num_ratings in ratings_count.items():
        if num_ratings < cutoff:
            remove_entries.append(entry)

    df = df.loc[~ df[attr].isin(remove_entries)]
    return df

# Remove insignificant entries


def remove_songs_users_below_rating(df, attrs):
    for attr in attrs:
        ratings = ratings_count(df, attr)
        cutoff = attrs[attr]
        remove_entries(df, cutoff, attr, ratings)
    return df_final


# We want our song to be listened by at least 120 users to be considered
# We want our users to have listened at least 90 songs
df_final = remove_songs_users_below_rating(df, {'user_id': 90, 'song_id': 120})

'''
Recommend Popular Songs
'''
# Build the function to find top n songs
# It gives top n songs among those being watched for more than min_interactions


def top_n_songs(data, n, attr, sortby, min_interaction=100,):

    # Finding songs with interactions greater than the minimum number of interactions
    recommendations = data[data[attr] > min_interaction]

    # Sorting values with respect to the average rating
    recommendations = recommendations.sort_values(by=sortby, ascending=False)

    return recommendations.index[:n]


def popular_songs(df, n, min_interactions=100):
    # Calculating average play_count
    average_count = df.groupby('song_id').mean()['play_count']

    # Calculating the frequency a song is played
    play_freq = df.groupby('song_id').count()['play_count']

    # Making a data frame with the average_count and play_freq
    final_play = pd.DataFrame(
        {'avg_count': average_count, 'play_freq': play_freq})

    return top_n_songs(df, n, 'play_freq', 'avg_count')


'''
Split Data
'''


def split_data(df, attrs):
    # Instantiating Reader scale with expected rating scale
    reader = Reader(rating_scale=(0, 5))  # use rating scale (0, 5)

    # Loading the dataset
    # Take only attributes wanted
    data = Dataset.load_from_df(
        df[attrs], reader)

    # Splitting the data into train and test dataset
    trainset, testset = train_test_split(
        data, test_size=0.4, random_state=42)  # Take test_size = 0.4
    return trainset, testset, data


trainset, testset, data = split_data(df, ["user_id", "song_id", "play_count"])
'''
Parameter Tuning
'''


def parameter_tune_KNN(data, k, mink_k, sims, user_based, min_support, measure):
    # Setting up parameter grid to tune the hyper parameters
    param_grid = {'k': k, 'min_k': mink_k,
                  'sim_options': {'name': sims,
                                  'user_based': [user_based], "min_support": min_support}
                  }

    # Performing 3-fold cross-validation to tune the hyper parameters
    gs = GridSearchCV(KNNBasic, param_grid, measures=[
                      measure], cv=3, n_jobs=-1)

    # Fitting the data
    gs.fit(data)

    opt_parameters = gs.best_params[measure]

    return opt_parameters.k, opt_parameters.min_k, opt_parameters['sim_options'].name, opt_parameters['sim_options'].user_based, opt_parameters['sim_options'].min_support


k, min_k, name, user_based, min_support = parameter_tune_KNN(data, k=[30, 40, 50], min_k=[3, 6, 9], sims=[
    'msd', 'cosine', 'pearson', "pearson_baseline"], user_based=True, min_support=[2, 4], measure='rmse')

'''
Train Model w/ Hyper Parameters
'''
# Train the best model found in above grid search
# Using the optimal similarity measure for user-user collaborative filtering
def train_model_user_user_opt(k, min_k, name, user_based, min_support, trainset):
    sim_options = {'name': name,
                'user_based': user_based,
                'min_support': min_support}

    # Creating an instance of KNNBasic with optimal hyperparameter values
    sim_user_user_optimized = KNNBasic(
        sim_options=sim_options, k=k, min_k=min_k, random_state=1, verbose=False)

    # Training the algorithm on the train set
    sim_user_user_optimized.fit(trainset)
    
    return sim_user_user_optimized

sim_user_user_optimized = train_model_user_user_opt(k, min_k, name, user_based, min_support, trainset)
'''
Model Performance
'''
# The function to calculate the RMSE, precision@k, recall@k, and F_1 score


def precision_recall_at_k(model, k=30, threshold=1.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)

    # Making predictions on the test data
    predictions = model.test(testset)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[: k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[: k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set Precision to 0 when n_rec_k is 0

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set Recall to 0 when n_rel is 0

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # Mean of all the predicted precisions are calculated
    precision = round(
        (sum(prec for prec in precisions.values()) / len(precisions)), 3)

    # Mean of all the predicted recalls are calculated
    recall = round((sum(rec for rec in recalls.values()) / len(recalls)), 3)

    accuracy.rmse(predictions)

    # Command to print the overall precision
    print('Precision: ', precision)

    # Command to print the overall recall
    print('Recall: ', recall)

    # Formula to compute the F-1 score
    print('F_1 score: ', round((2 * precision * recall) / (precision + recall), 3))


'''
Results Return
'''


def get_recommendations(data, user_id, top_n, algo):

    # Creating an empty list to store the recommended product ids
    recommendations = []

    # Creating an user item interactions matrix
    user_item_interactions_matrix = data.pivot_table(
        index='user_id', columns='song_id', values='play_count')

    # Extracting those business ids which the user_id has not visited yet
    non_interacted_products = user_item_interactions_matrix.loc[user_id][
        user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    # Looping through each of the business ids which user_id has not interacted yet
    for item_id in non_interacted_products:

        # Predicting the ratings for those non visited restaurant ids by this user
        est = algo.predict(user_id, item_id).est

        # Appending the predicted ratings
        recommendations.append((item_id, est))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Returning top n highest predicted rating products for this user
    return recommendations[:top_n]


recommendations = get_recommendations(
    df_final, 6958, 5, sim_user_user_optimized)
pd.DataFrame(recommendations, columns=['song_id', 'predicted_play_count'])

# Let us compute precision@k, recall@k, and F_1 score with k = 10
precision_recall_at_k(sim_user_user_optimized)

# # Recommend top 10 songs using the function defined above
# list(top_n_songs(final_play, 10, 200))
