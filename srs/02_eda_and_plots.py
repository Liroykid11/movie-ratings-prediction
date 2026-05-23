import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Arseni\Documents\movie-ratings-prediction\movie-ratings-prediction\data\processed\cleaned_movies_v1.csv')
correlations = df.corr()['Vote_Average'].sort_values(ascending=False)
correlations = correlations[correlations.index != 'Vote_Average']


plt.figure(figsize=(10, 8))
plt.barh(correlations.index, correlations.values)
plt.title('Feature Correlations with Movie Rating')
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('results/eda/correlation_plot.png', dpi=150, bbox_inches='tight')
plt.close() 

# Weak correlation detected (max |r| = 0.18) 
# Linear regression may not perform well with current feature

plt.hist(df['Vote_Average'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Movie Ratings', fontsize=14)
plt.xlabel('Vote Average', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.savefig('results/eda/vote_average_distribution.png', dpi = 150)
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(df['Vote_Average'])
plt.title('Boxplot of Movie Ratings - Outlier Detection')
plt.ylabel('Vote Average')
plt.savefig('results/eda/ratings_boxplot.png', dpi = 150)
plt.close()

outliers_0 = df[df['Vote_Average'] < 0.1]  # count - 100
outliers_10 = df[df['Vote_Average'] > 9.9]  # count - 1
df = df[df['Vote_Average'] <= 9.9]
df = df[df['Vote_Average'] >= 0.1]

genre_cols = [col for col in df.columns if col.startswith('genre_')]

genre_ratings = {}
for col in genre_cols:
    genre_name = col.replace('genre_', '')
    rating = df[df[col] == 1]['Vote_Average'].mean()
    genre_ratings[genre_name] = rating

genre_ratings = pd.Series(genre_ratings).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
genre_ratings.head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Genres by Average Rating')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.savefig('results/eda/genre_ratings.png', dpi=150)
plt.close()


plt.figure(figsize=(10, 6))
plt.boxplot(df['year'])
plt.title('Boxplot of Movie Release Years')
plt.ylabel('Year')
plt.tight_layout()
plt.savefig('results/eda/year_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

df = df[df['year']>1920]

yearly_rating = df.groupby('year')['Vote_Average'].mean()

plt.figure(figsize=(12, 6))
plt.plot(yearly_rating.index, yearly_rating.values, marker='o', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.title('Movie Rating Trend Over Years')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/eda/yearly_trend.png', dpi=150)
plt.close()

df.to_csv('data/processed/cleaned_movies_v2.csv', index=False)


