#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time


def load_data(filepath: str) -> np.ndarray:
    """
    load a user-movie-rating array from a .npy file

    Parameters
        filepath (str): path to the .npy file

    Returns
        np.ndarray: 2D array of shape (n_samples, 3) with [user_id, movie_id, rating]
    """
    # check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    # load data and raise error if data is not as expected
    data = np.load(filepath, mmap_mode='r') # mmap_mode='r' for large files
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Expected a 2D array with 3 columns: user_id, movie_id, rating")
    return data


def transform_data(data: np.ndarray):
    """
    transform a numpy 2d array to a COO sparse matrix. Transform it to CSR matrix for efficiency.

    Parameters
        data (np.ndarray): 2D array of shape (n_samples, 3) with [user_id, movie_id, rating]

    Returns
        csr_matrix: CSR sparse matrix of shape (n_users, n_movies) with True where a user rated a movie
        int: total number of users
        int: total number of movies
        
    """

    # parse columns
    users = data[:, 0].astype(dtype=int)
    movies = data[:, 1].astype(dtype=int)

    n_users = users.max()
    n_movies = movies.max()

    #scipy uses ID's starting from 0
    users -= 1
    movies -= 1

    # Create sparse matrix (ratings irrelevant, just presence)
    #! this can be experimented with later
    data_values = np.ones_like(users, dtype=np.uint8)
    # use coo_matrix to create sparse matrix
    coo = coo_matrix((data_values, (users, movies)), shape=(n_users, n_movies), dtype=bool)
    del users, movies, data # free memory
    # use csr format for efficient row slicing
    csr = coo.tocsr()
    del coo # free memory
    return csr, n_users, n_movies


def csr_to_user_movie_lists(csr:csr_matrix, n_users:int) -> list:
    """
    create list of arrays, each array containing the movie indices rated by that user.

    Parameters
        csr (csr_matrix): CSR sparse matrix of shape (n_users, n_movies) with True where a user rated a movie
        n_users (int): total number of users

    Returns
        list: list of arrays, each array containing the movie indices rated by that user
    """

    # create list of arrays, each array contains the movie indices rated by that user
    user_movie_lists = [
        csr.indices[csr.indptr[i]:csr.indptr[i+1]]
        for i in range(n_users)
    ]
    return user_movie_lists


def create_signatures(user_movie_lists:list[np.ndarray], n_users:int, n_movies:int, k:int) -> np.ndarray:
    """
    compute minhash signatures for all users.

    Parameters
        user_movie_lists (list[np.ndarray]): CSR sparse matrix of shape (n_users, n_movies) with True where a user rated a movie
        n_users (int): total number of users
        n_movies (int): total number of movies
        k (int): number of permutations

    Returns
        np.ndarray: MinHash signature matrix of shape (n_users, k)
    """

    # create k random permutations of movie indices
    permutations = np.array([
        np.random.permutation(n_movies)
        for a in range(k)
    ], dtype=np.int32)

    # initialize signature matrix
    signatures = np.empty(shape=(n_users, k), dtype=np.int32)

    # loop over each permutation and compute minhash signatures
    for j in range(k): 
        perm = permutations[j]
        # compute: min perm[movie] for each user
        signatures[:, j] = np.array([
            perm[movies].min() # mishash value for this user and this permutation
            for movies in user_movie_lists
        ])
    return signatures


def split_signatures_into_bands(signatures:np.ndarray, bands:int, rows:int, n_users:int) -> np.ndarray:
    """
    split signatures into bands.

    Parameters
        signatures (np.ndarray): signature matrix of shape (n_users, k)
        bands (int): number of bands
        rows (int): number of rows per band
        n_users (int): total number of users

    Returns
        np.ndarray: Array of shape (n_users, bands, rows) containing banded signatures.
    """

    banded_signatures = signatures.reshape(n_users, bands, rows)
    return banded_signatures


def put_users_in_buckets(banded_signatures:np.ndarray, rows:int, bands:int) -> list:
    """
    hash users into buckets based on their banded MinHash signatures.

    Parameters
        banded_signatures (np.ndarray): array of shape (n_users, bands, rows)
        rows (int): number of rows per band
        bands (int): number of bands

    Returns
        list: a list of length `bands`, where each element is a list of buckets
              each bucket is a numpy array of user IDs that share the same band hash
    """

    # we choose a random hash multiplier per row (for stable hashing)
    # this mixes the row values into a single integer without collisions being catastrophic.
    multipliers = np.random.randint(low=1, high=2**31 - 1, size=rows, dtype=np.int64)

    band_buckets = []   # list of: list of buckets, each bucket is a numpy array of user IDs

    for b in range(bands):

        # shape (n_users, rows)
        band = banded_signatures[:, b, :]

        # compute hash per user:
        band_hashes = (band.astype(np.int64) * multipliers).sum(axis=1)

        # sort users by hash so equal-hash users become adjacent
        order = np.argsort(band_hashes)
        sorted_hashes = band_hashes[order]

        # find boundaries where the hash value changes
        changes = np.where(sorted_hashes[1:] != sorted_hashes[:-1])[0] + 1

        # split into groups (buckets)
        groups = np.split(order, changes)

        # keep buckets only if they contain at least 2 users
        buckets = [g for g in groups if len(g) >= 2]

        band_buckets.append(buckets)
    return band_buckets


def create_candidate_pairs(band_buckets:list) -> np.ndarray:
    """
    create candidate user pairs from LSH buckets.

    Parameters
        band_buckets (list): list of bands, where each band contains
            buckets of user IDs

    Returns
        np.ndarray: array of shape (n_pairs, 2) with unique user ID pairs
    """

    candidate_pairs = set()

    # loop through all buckets
    for buckets in band_buckets:
        for bucket in buckets:
            bucket = np.asarray(bucket)
            bucket_size = bucket.size
            if bucket_size < 2 or bucket_size > 50: #! this cut off can be experimented with
                continue
            # generate all unordered pairs inside the bucket
            i = np.repeat(np.arange(bucket_size - 1), np.arange(bucket_size - 1, 0, -1))
            j = np.concatenate([np.arange(x + 1, bucket_size) for x in range(bucket_size - 1)])
            pairs = np.stack((bucket[i], bucket[j]), axis=1)

            for u, v in pairs:
                if u < v: # makes it so the other combination of pairs, will not be again added to the set
                    candidate_pairs.add((u, v))

    # convert set to numpy array
    candidate_pairs = np.array(list(candidate_pairs), dtype=np.int32)
    return candidate_pairs

    
def jaccard_similarity(u_items:np.ndarray, v_items:np.ndarray) -> float:
    """
    compute the Jaccard similarity between two sets of movie indices.

    Parameters
        u_items (np.ndarray): sorted unique movie indices for user u.
        v_items (np.ndarray): sorted unique movie indices for user v.

    Returns
        float: Jaccard similarity between the two users.
    """

    # intersection size
    intersection_size = np.intersect1d(u_items, v_items, assume_unique=True).size
    # union and size
    union = np.union1d(u_items, v_items)
    union_size = len(union)

    return intersection_size / union_size if union_size > 0 else 0.0


def filter_users_on_jaccard(candidate_pairs:np.ndarray, user_movie_lists:list, threshold:float, file_name:str):
    """
    filter candidate user pairs by Jaccard similarity and write matches to a file

    Parameters
        candidate_pairs (np.ndarray): array of (u, v) user ID pairs
        user_movie_lists (list): list of movie index arrays for each user
        threshold (float): minimum Jaccard similarity required
        file_name (str): path to the output file where matching pairs are appended

    Returns
        None
    """

    for u, v in candidate_pairs:
        sim = jaccard_similarity(user_movie_lists[u], user_movie_lists[v])
        if sim > threshold:
            # open and close file when new pair is found folowing the hint in the assignment
            with open(file_name, "a") as f:
                # write u1,u2 to file
                f.write(f"{u},{v}\n")


#! this function can be removed before submission
def write_log_file(output_file, seed, k, bands, rows, threshold, time):
    print("at write_log_file")

    with open(output_file, "r") as f:
        matches =  sum(1 for _ in f)  # count each line to get number of pairs

    with open('runs.txt', 'a') as f:
        f.write(f"seed={seed}, k={k}, bands={bands}, rows={rows}, threshold={threshold}, time={time:.2f}s, matches={matches}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Find similar user pairs using LSH on Netflix data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/user_movie_rating.npy',
        help='Path to input .npy file (default: user_movie_rating.npy)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='result.txt',
        help='Path to output file (default: result.txt)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Jaccard similarity threshold (default: 0.5)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=120,
        help='Number of hash functions (default: 120)'
    )
    parser.add_argument(
        '--bands',
        type=int,
        default=20,
        help='Number of bands (default: 10, giving 12 rows per band)'
    )

    args = parser.parse_args()

    # Validate parameters
    if args.k % args.bands != 0:
        print(f"Error: k ({args.k}) must be divisible by bands ({args.bands})")
        sys.exit(1)



    # set the seed
    np.random.seed(args.seed)
    
    # set some variables
    user_movie_rating_path = args.input
    k = args.k # number of hash functions (permutations)
    bands = args.bands
    rows = k // bands
    output_file = args.output
    threshold = args.threshold  # Jaccard similarity threshold

    #! this can be removed before submission
    start_time = time.time()

    # load and tranform data
    data = load_data(filepath=user_movie_rating_path)
    data, n_users, n_movies = transform_data(data)

    # create list of arrays, each array contains the movie indices rated by that user
    user_movie_lists = csr_to_user_movie_lists(data, n_users)

    # create signatures
    signatures = create_signatures(user_movie_lists, n_users, n_movies, k)
    
    # split signatures into bands and put users with identical band signatures into the same bucket
    banded_signatures = split_signatures_into_bands(signatures, bands, rows, n_users)
    band_buckets = put_users_in_buckets(banded_signatures, rows, bands)

    # create pairs of users that are similar candidates
    candidate_pairs = create_candidate_pairs(band_buckets)

    # create or clear output file to avvoind appending to an already exsisting one
    open(output_file, "w").close()

    # calculate the jaccard simmilarity for the similar candidates. Write all user pairs to text file with score above threshold
    filter_users_on_jaccard(candidate_pairs=candidate_pairs, user_movie_lists=user_movie_lists, threshold=threshold, file_name = output_file)

    #! this (3 lines)can be removed before submission
    total_time = time.time() - start_time
    # append to log file
    write_log_file(output_file, args.seed, k, bands, rows, threshold, total_time, )

if __name__ == "__main__":
    main()



