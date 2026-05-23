import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os
import joblib

def language(language):
    if language == 'en':
        return 'en'
    elif language == 'ja':
        return 'ja'
    elif language == 'es':
        return 'es'
    elif language == 'fr':
        return 'fr'
    elif language == 'ko':
        return 'ko'
    elif language == 'zh':
        return 'zh'
    elif language == 'it':
        return 'it'
    elif language == 'cn':
        return 'cn'
    elif language == 'ru':
        return 'ru'
    elif language == 'de':
        return 'de'
    else:
        return 'other'

df = pd.read_csv(r'C:\Users\Arseni\Downloads\mymoviedb.csv', engine='python')
df = df.drop('Title', axis = 1)
df['Release_Date'] = df['Release_Date'].str.strip()
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df = df.dropna(subset=['Release_Date'])
df['year'] = df['Release_Date'].dt.year
df['month'] = df['Release_Date'].dt.month
df['day_of_week'] = df['Release_Date'].dt.day_of_week
df = df.drop('Release_Date', axis = 1)
df = df.drop('Overview', axis = 1)
df = df.drop('Poster_Url', axis = 1)
df['Original_Language'] = df['Original_Language'].apply(language)
df = pd.get_dummies(df, columns=['Original_Language'], drop_first=True)
index_null = df[df['Genre'].isnull()].index.tolist()
df = df.drop(index = index_null, axis = 0)

df['Genre_list'] = df['Genre'].str.split(', ')
all_genres = df['Genre_list'].explode()
top_15_genres = all_genres.value_counts().head(15).index.tolist()

df['other_genre'] = df['Genre_list'].apply(
    lambda x: 1 if any(genre not in top_15_genres for genre in x) else 0
)

df['Genre_list_filtered'] = df['Genre_list'].apply(
    lambda x: [g for g in x if g in top_15_genres]
)

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genre_list_filtered'])
genre_dummies = pd.DataFrame(
    genre_encoded,
    columns=[f'genre_{genre}' for genre in mlb.classes_],
    index=df.index
)
df = pd.concat([df, genre_dummies], axis=1)
df = df.drop(['Genre', 'Genre_list', 'Genre_list_filtered'], axis=1)


df.to_csv('data/processed/cleaned_movies.csv', index=False)