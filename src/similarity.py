import numpy as np

def similarity(user1_movies: np.array, user2_movies: np.array) -> float:
    """
    Compute Jaccard similarity between two users based on items they rated.
    We only care about movies that were rated but not the actual ratings
    """

    intersection = np.intersect1d(user1_movies, user2_movies)
    union = np.union1d(user1_movies, user2_movies)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)
