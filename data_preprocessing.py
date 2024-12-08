import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Encode labels
    label_encoder = LabelEncoder()
    df['genre_label'] = label_encoder.fit_transform(df['genre'])

    # Vectorize lyrics
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['lyrics'])
    y = df['genre_label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=23, shuffle=True)

    # Reset indices for labels
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, vectorizer
