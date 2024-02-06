import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import csv

def music_recommender(songs_df, user_preferences):
    # Perform data cleaning and preprocessing
    # Here, you can modify the code to clean and preprocess your specific data
    # For example, you may need to drop NaN values, remove duplicates, or convert categorical data to numerical data

    songs_df1 = songs_df.drop(['artist_name', 'track_id', 'track_name'], axis=1, inplace=False)
    print(songs_df.head())

    user_preferences = user_preferences.drop(
        ['artist_name', 'track_id', 'track_name'], axis=1, inplace=False)
    print(songs_df1.head())

    # Use k-means clustering to cluster the songs into groups based on their musical features
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(songs_df1)

    # Normalize user preferences using MinMaxScaler
    scaler = MinMaxScaler()
    user_preferences_normalized = scaler.fit_transform(user_preferences)


    # Generate a playlist of recommended songs for the user
    user_cluster = kmeans.predict(user_preferences_normalized)
    print(user_cluster.shape)
    print(songs_df['track_id'].shape)
    print(songs_df1.columns)
    print(clusters)

    # Compute cluster assignments for each song
    song_clusters = songs_df.groupby('song_id')['cluster'].apply(
    lambda x: x.iloc[0]).to_dict()

    # Create a mapping of song to cluster
    song_cluster_mapping = {}
    for song_id in song_clusters:
        cluster = song_clusters[song_id]
        song_cluster_mapping[song_id] = [cluster] * \
            len(songs_df[songs_df['song_id'] == song_id])



    # Use the mapping to add a 'cluster' column to songs_df
    songs_df['cluster'] = [item for sublist in song_cluster_mapping.values()
                        for item in sublist]

    # Find the cluster of the user's preferences
    user_cluster = user_preferences['cluster'].mode()[0]

    # Find songs belonging to the user's preferred cluster
    recommended_songs = songs_df[songs_df['cluster'] == user_cluster]

    # Display the recommended songs
    print(recommended_songs)

    # Filter the original songs DataFrame based on the predicted cluster
    recommended_songs = songs_df[songs_df1['cluster'] == user_cluster]
        
    # Write the recommended songs to a CSV file
    recommended_songs.to_csv('recommended_songs.csv', index=False)


def main():

    # Read in user preferences from a CSV file
    # Replace with your CSV file name
    filename = r"D:\Projects\Recommandation System\user_Prefrence.csv"
    user_preferences_df = pd.read_csv(
        filename,  dtype={'track_name': 'str', 'artist_name': 'str'})

    # Call the music_recommender function to generate recommended playlists
    songs_df = pd.read_csv(
        'D:\Projects\Recommandation System\SpotifyAudioFeaturesApril2019.csv',  dtype={'track_name': 'str', 'artist_name': 'str'})
    music_recommender(songs_df, user_preferences_df)


if __name__ == '__main__':
    main()
