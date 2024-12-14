import os
import dash
import numpy as np
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Load data
movies = pd.read_csv('https://liangfgithub.github.io/MovieData/movies.dat?raw=true', sep='::', engine='python',
                     encoding="ISO-8859-1", header=None)
movies.columns = ['MovieID', 'Title', 'Genres']
ratings = pd.read_csv('https://liangfgithub.github.io/MovieData/ratings.dat?raw=true', sep='::', engine='python',
                      header=None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
# Get movie stats and display movies for rating
movie_stats = (
    ratings.groupby('MovieID')
    .agg(AvgRating=('Rating', 'mean'), RatingCount=('Rating', 'size'))
    .reset_index()
)
pop_movies = movie_stats[movie_stats['RatingCount'] > 500]
pop_movies = pd.merge(pop_movies, movies, on='MovieID')
selected_movies = pop_movies.sort_values(by='AvgRating', ascending=False).head(100)


# Function for generating recommendations (same as in your code)
def myIBCF(newuser):
    S_mat = pd.read_csv("https://github.com/Baizhao-666/CS598Material/blob/main/Smat.csv?raw=true", index_col=0)

    tmp_user = pd.Series(np.nan, index=S_mat.index)
    for idx in newuser.index:
        if newuser.loc[idx] > 0:
            tmp_user.loc[idx] = newuser.loc[idx]
    newuser = tmp_user
    predictions = pd.Series(index=S_mat.index, dtype=float)

    for i in S_mat.index:
        S_i = S_mat.loc[i].dropna()

        rated_movies = S_i.index[newuser[S_i.index].notna()]
        S_i_rated = S_i[rated_movies]

        numerator = np.sum(S_i_rated * newuser[rated_movies])
        denominator = np.sum(S_i_rated)

        if denominator > 0:
            predictions[i] = numerator / denominator

    predictions[newuser.notna()] = np.nan
    predictions = predictions.sort_values(ascending=False).dropna()

    if predictions.size < 10:
        res = predictions.index.tolist()
        popular_movies = pd.read_csv(
            'https://github.com/Baizhao-666/CS598Material/blob/main/popular_movies.csv?raw=true')
        for movie in popular_movies["MovieID"]:
            if movie not in res and len(res) <= 10:
                res.append(f"m{movie}")
    else:
        top_indices = predictions.nlargest(10).index
        res = top_indices.tolist()

    return res


# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    # Step 1: Movie Rating
    html.Div(id='step1', children=[
        html.Div(
            children=[
                html.H3('Step 1: Rate as many movies as possible',
                        style={'display': 'inline-block', 'margin-right': '20px'}),
                html.Button('Submit Ratings', id='submit-button', n_clicks=0, style={
                    'display': 'inline-block',
                    'background-color': '#28A745',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '5px',
                    'padding': '10px 20px',
                    'font-size': '16px',
                    'cursor': 'pointer',
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                    'transition': 'background-color 0.3s ease, transform 0.2s ease',
                    'text-align': 'center',
                    'outline': 'none',
                    'margin-left': '20px',
                    'margin-top': '10px'
                })
            ],
            style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),
        html.P('Rate the movies displayed below. The more movies you rate, the better your recommendations will be.'),
        html.Div(id='movies-grid', children=[],
                 style={'max-height': '400px', 'overflow-y': 'scroll', 'padding': '10px', 'border': '2px solid #007BFF',
                        'border-radius': '5px', 'background-color': '#F0F8FF'}),
        html.Div(id='rating-output')
    ], style={'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '5px', 'background-color': '#E9F7FF'}),

    # Step 2: Recommended Movies
    html.Div(id='step2', children=[
        html.Div(
            html.H3('Step 2: Your Recommended Movies',
                    style={'display': 'inline-block', 'margin-right': '20px'}),
            style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'position': 'sticky',
                   'top': '0', 'background-color': '#E9F7FF', 'z-index': '100'}
        ),
        html.P('Here are the movies we recommend based on your ratings. Enjoy!'),
        html.Div(id='recommended-movies', children=[])
    ], style={'border': '2px solid #28A745', 'padding': '20px', 'border-radius': '5px', 'background-color': '#E9F9EE'})
])


# Callback to generate the movie grid for ratings
@app.callback(
    Output('movies-grid', 'children'),
    Output('submit-button', 'disabled'),
    Input('submit-button', 'n_clicks'),
    State('movies-grid', 'children')
)
def display_movie_grid(n_clicks, current_children):
    children = []
    for _, movie in selected_movies.iterrows():
        children.append(html.Div([
            html.Img(src=f"https://liangfgithub.github.io/MovieImages/{movie['MovieID']}.jpg",
                     style={'width': '80%', 'height': '80%'}),
            html.H5(movie['Title'].strip()),
            dcc.Slider(
                id=f"m{movie['MovieID']}",
                min=0,
                max=5,
                step=1,
                value=0,
                marks={i: str(i) for i in range(6)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'display': 'inline-block', 'width': '18%', 'padding': '10px', 'text-align': 'center'}))

    return children, False


# Callback to submit ratings and generate recommendations
@app.callback(
    Output('rating-output', 'children'),
    Output('recommended-movies', 'children'),
    Input('submit-button', 'n_clicks'),
    State('movies-grid', 'children')
)
def submit_ratings_and_recommend(n_clicks, movie_elements):
    if n_clicks > 0:
        ratings_dict = {}
        for movie_element in movie_elements:
            movie_id = movie_element['props']['children'][2]['props']['id']
            rating = movie_element['props']['children'][2]['props']['value']
            ratings_dict[movie_id] = rating

        # Generate recommendations based on ratings
        new_user = pd.Series(ratings_dict)
        recommended_movies = myIBCF(new_user)

        # Create the recommended movies grid
        rec_children = []
        for movie_id in recommended_movies:
            movie_id = int(movie_id.replace('m', ''))
            movie = movies[movies['MovieID'] == movie_id].iloc[0]
            rec_children.append(html.Div([
                html.Img(src=f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg", style={'width': '100%'}),
                html.H5(movie['Title'].strip())
            ], style={'display': 'inline-block', 'width': '18%', 'padding': '10px'}))

        return "Ratings submitted successfully! Scroll down to see your recommendations.", rec_children

    return "", []


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
