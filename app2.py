import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Goodreads Book Analysis')

st.write('Loading dataset...')
df = pd.read_csv('goodreads.csv')

st.write('Dataset loaded')
st.write('Dataset', df.head())


df.drop_duplicates(inplace=True)


st.write('Visualizing rating distribution...')
fig, ax = plt.subplots()
sns.histplot(df['Rating'], bins=20, kde=False, ax=ax,  color='skyblue')
ax.set_title('Rating Distribution')
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
st.pyplot(fig)

df_counts = pd.read_csv('Ratings_count.csv')
df_counts.rename(columns={'Book_titles': 'Book_Title'}, inplace=True)
df_counts['Book_Title'] = df_counts['Book_Title'].str.split('(').str[0].str.strip()

df = df[df['Genres'].str.len() != 2]
df['Genres'] = df['Genres'].str.replace('[', '').str.replace(']', '').str.replace("'", '').str.split(', ')
df3 = df.explode('Genres')

selected_genres = df3['Genres'].value_counts().index[:20]
df4 = df3[df3['Genres'].isin(selected_genres)]
df5 = df4.groupby('Book_Title').agg({
    'Genres': list,
    'Rating': 'mean'
}).reset_index()

def has_genre(genres, genre):
    return 1 if genre in genres else 0

for genre in selected_genres:
    df5[genre] = df5['Genres'].apply(lambda x: has_genre(x, genre))

df6 = df5.merge(df_counts, on='Book_Title', how='inner')

st.write('Visualizing genres distribution...')
genres_count = df4['Genres'].value_counts()
fig, ax = plt.subplots(figsize=(24, 6))
sns.barplot(x=genres_count.index, y=genres_count.values, ax=ax, palette='Set2')
ax.set_title('Genres Distribution')
st.pyplot(fig)

st.write('Top 10 Books by Rating Count')
most_rated = df6.sort_values('Ratings_count', ascending=False).head(10).set_index('Book_Title')
fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=most_rated['Ratings_count'], y=most_rated.index, palette='Set2', ax=ax)
ax.set_xlabel('Rating Count')
ax.set_ylabel('Book Title')
ax.set_title('Top 10 Books by Rating Count')
st.pyplot(fig)

st.write('Top 10 Authors with Highly Rated Books')
high_rated_author = df[df['Rating'] >= 4.3]
high_rated_author = high_rated_author.groupby('Author')['Book_Title'].count().reset_index().sort_values('Book_Title', ascending=False).head(10).set_index('Author')
fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=high_rated_author['Book_Title'], y=high_rated_author.index, palette='Set2', ax=ax)
ax.set_xlabel('Number of Books')
ax.set_ylabel('Authors')
ax.set_title('Top 10 Authors with Highly Rated Books')
st.pyplot(fig)


random_input = {
    'Rating': 4.25,
    'Ratings_count': 15342,
    'Fiction': 1,
    'Fantasy': 0,
    'Young Adult': 1,
    'Classics': 0,
    'Historical': 0,
    'Romance': 1,
    'Science Fiction': 0,
    'Adventure': 0,
    'Nonfiction': 0,
    'Contemporary': 1,
    'Mystery': 0,
    'Thriller': 0,
    'Memoir': 0,
    'Biography': 0,
    'Horror': 0,
    'Self Help': 0,
    'Graphic Novels': 0,
    'Short Stories': 0,
    'Science': 0
}
