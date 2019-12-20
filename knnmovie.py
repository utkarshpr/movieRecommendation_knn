import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
ds = pd.read_csv("movies.csv")
rt=pd.read_csv("ratings.csv")
from scipy.sparse import csr_matrix
# pivot ratings into movie features
df_movie_features = rt.pivot(index='movieId',columns='userId',values='rating').fillna(0)
movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(ds.set_index('movieId').loc[df_movie_features.index].title))
}
# convert dataframe of movie features to scipy sparse matrix
mat_movie_features = csr_matrix(df_movie_features.values)

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(mat_movie_features)

def matching(mapper,fav_movies):
    match_tuple = []
    for title, idx in mapper.items():
        if title == fav_movies:
            match_tuple.append((title,idx))
    return match_tuple[0][1]
def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = matching(mapper,fav_movie)    
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))
my_favorite = 'Iron Man (2008)'

make_recommendation(
    model_knn=model_knn,
    data=mat_movie_features,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)        