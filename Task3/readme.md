# Report: Exploring Top 50 Spotify Songs - 2019

## Introduction:
The Top 50 Spotify Songs - 2019 dataset contains information on the most popular songs on Spotify in 2019. In this report, we will explore the data to gain insights into what makes a successful song and which genres are currently popular.

## Data Exploration:
We began by exploring the distribution of the variables in the dataset. We found that the majority of the songs were in the pop genre and had high energy, danceability, and valence. The popularity of the songs varied widely within each genre.

## Feature Engineering:
We created two new variables: 'tempo' which is the number of beats per second, and 'artist_count' which is the number of artists on the track. These variables helped us understand the relationship between tempo and energy and the prevalence of collaboration among artists.

## Feature Selection:
We selected a subset of the variables for further analysis, including 'track_name', 'artist_name', 'genre', 'tempo', 'energy', 'danceability', 'loudness', 'liveness', 'valence', 'length', 'acousticness', 'speechiness', 'popularity', and 'artist_count'.

## Visualizations:
We created several visualizations to explore the relationships between variables. Scatterplot matrices and scatterplots colored by genre helped us identify correlations between variables, while violin plots and bar charts helped us compare variables across genres.

## Findings:

- Popularity is positively correlated with energy, danceability, and valence, but negatively correlated with acousticness and length.
- The majority of the top 50 songs in 2019 were in the pop genre, but reggaeton and trap had the highest average popularity.
- There is a positive relationship between tempo and energy, with electronic and reggaeton songs having the highest tempos.
- There is no clear relationship between length and energy, but hip hop and latin songs tend to be longer than other genres.



## Conclusion:
Our analysis suggests that successful songs tend to be high energy, danceable, and have a positive emotional tone. Collaboration among artists is also prevalent in popular songs. Pop, reggaeton, and trap are currently popular genres, but the popularity of individual songs within each genre can vary widely.