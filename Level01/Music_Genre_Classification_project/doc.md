# Machine Learning Classification Project

This machine learning classification project involves building and evaluating various algorithms to predict the genre of music tracks based on a set of features. The dataset consists of a training set with 14,395 rows and 18 columns, and a test set with 3,600 rows and 17 columns. The target variable to be predicted is 'Class', representing the genre of the music track.

## Dataset Description

### Training Dataset

The training dataset (`train.csv`) contains the following columns:

1. `artist name`: Name of the artist who performed the track.
2. `track name`: Name of the music track.
3. `popularity`: Popularity score of the track, indicating its popularity level.
4. `danceability`: A measure of how suitable the track is for dancing based on musical elements such as tempo and rhythm.
5. `energy`: A measure of the intensity and activity of the track.
6. `key`: The key of the track in pitch class notation.
7. `loudness`: The overall loudness of the track in decibels (dB).
8. `mode`: Indicates whether the track is in a major or minor key (1 for major, 0 for minor).
9. `speechiness`: Detects the presence of spoken words in the track.
10. `acousticness`: A measure of whether the track is acoustic (1.0 for high confidence, 0.0 for low confidence).
11. `instrumentalness`: Predicts whether the track contains no vocals.
12. `liveness`: Detects the presence of an audience in the recording, indicating whether the track was performed live.
13. `valence`: A measure of the musical positiveness conveyed by the track.
14. `tempo`: The overall estimated tempo of the track in beats per minute (BPM).
15. `duration in milliseconds`: The duration of the track in milliseconds.
16. `time_signature`: The time signature used in the track's notation to specify the number of beats per measure.
17. `Class`: The target variable representing the genre of the track, such as Rock, Indie, Alt, Pop, Metal, HipHop, Alt_Music, Blues, Acoustic/Folk, Instrumental, Country, and Bollywood.

### Test Dataset

The test dataset (`test.csv`) contains the same columns as the training dataset, except for the absence of the target variable `Class`.

## Data Preprocessing

Before training the machine learning models, the data undergoes several preprocessing steps:

```python
# Load the data
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and target variable
X = train_data.drop(['Class'], axis=1)
y = train_data['Class']

# Handling Missing Values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_imputed[:, [0, 1]])  # One-hot encode 'artist name' and 'track name'

# Concatenate scaled and encoded features
import numpy as np

X_processed = np.concatenate([X_scaled, X_encoded, X_imputed[:, 2:]], axis=1)


# Data Splitting
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

```

## Machine Learning Algorithms
The project involves training and evaluating the following machine learning algorithms:
```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ml_algorithms = [
    GradientBoostingClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(kernel='rbf')
]

```

### Model Training and Evaluation

```python
# Train and evaluate machine learning algorithms
algorithm_accuracy = {}
for algorithm in ml_algorithms:
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    algorithm_accuracy[type(algorithm).__name__] = acc
    print(f"Machine Learning Algorithm: {type(algorithm).__name__} Accuracy: {acc}")
```
### Choosing the Best Algorithm

```python
# Find the algorithm with the highest accuracy
best_algorithm_name = max(algorithm_accuracy, key=algorithm_accuracy.get)
best_algorithm = ml_algorithms[best_algorithm_name]
print("Best Algorithm:", best_algorithm_name)

```
### Making Predictions

```python
# Preprocess the test data similar to the training data
test_imputed = imputer.transform(test_data)
test_scaled = scaler.transform(test_imputed)
test_encoded = encoder.transform(test_imputed[:, [0, 1]])
test_processed = np.concatenate([test_scaled, test_encoded, test_imputed[:, 2:]], axis=1)

# Make predictions on the test data using the best algorithm
best_algorithm.fit(X_processed, y)
best_algorithm_results = best_algorithm.predict(test_processed)
 ```

 ### Saving Predictions
```python
# Create a DataFrame for submission
submission_df = pd.DataFrame({'Id': test_data['Id'], 'Class': best_algorithm_results})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
print("Submission CSV file created: submission.csv")
```

### Confusion Matrix and Classification Report
For the chosen best algorithm (Logistic Regression), a confusion matrix is generated to analyze the model's performance on the validation data.


```python
# Generate and display the confusion matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_algorithm.classes_)
disp.plot(cmap='Blues', values_format='.0f')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
# Print the classification report
print("Classification Report for Logistic Regression:")
print(classification_report(y_val, y_pred))
```

