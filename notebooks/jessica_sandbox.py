import argparse
import numpy as np
import os
from scipy.sparse import coo_matrix
from tqdm import tqdm #! remove eventually because we can only use numpy and scipy libraries

parser = argparse.ArgumentParser(description="LSH user similarity")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# Set the seed
np.random.seed(args.seed)


# load user_movie_rating.npy and parse columns


# check if file exists
user_movie_rating_path = "user_movie_rating.npy"
if not os.path.exists(user_movie_rating_path):
    raise FileNotFoundError(f"{user_movie_rating_path} not found in current directory")

# load data and raise error if data is not as expected
data = np.load(user_movie_rating_path, mmap_mode='r') # mmap_mode='r' for large files
if data.ndim != 2 or data.shape[1] != 3:
    raise ValueError("Expected a 2D array with 3 columns: user_id, movie_id, rating")

# parse columns
users = data[:, 0].astype(int)
movies = data[:, 1].astype(int)
ratings = data[:, 2].astype(int)

#! at the end the specific rating does not matter for building the user-item matrix

n_users = users.max()
n_movies = movies.max()

print(f"Loaded {data.shape[0]} interactions")
print(f"Users: {n_users} ")
print(f"Movies: {n_movies} ")

# example: show first 10 raw rows and their mapped indices
print("First 10 raw rows (user_id, movie_id, rating):")
print(data[:10])


# create a sparse user-item rating matrix


#scipy uses ID's starting from 0
users -= 1
movies -= 1

data_values = np.ones_like(users, dtype=np.uint8) # rating presence indicator, ratings is ignored
coo = coo_matrix((data_values, (users, movies)), shape=(n_users, n_movies), dtype=bool)
print(f"Sparse rating matrix shape: {coo.shape}, nnz={coo.nnz}")
del users, movies, ratings, data
csr = coo.tocsr()
del coo

#print some csr data
print(f"CSR matrix shape: {csr.shape}, nnz={csr.nnz}")
print(f"CSR matrix data sample (first 10 entries): {csr.data[:40]}")

# create list of arrays, each array contains the movie indices rated by that user
user_movie_lists = [
    csr.indices[csr.indptr[i]:csr.indptr[i+1]]
    for i in range(n_users)
]


k = 120 # number of hash functions (permutations)
bands = 30
rows = k // bands # rows is now 4

# create k random permutations of movie indices
permutations = np.array([
    np.random.permutation(n_movies)
    for a in range(k)
], dtype=np.int32)

# initialize signature matrix
signatures = np.empty(shape=(n_users, k), dtype=np.int32)

# loop over each permutation and compute minhash signatures
for j in tqdm(range(k)): #! tqdm to show progress bar. remove in final version
    perm = permutations[j]
    # compute: min perm[movie] for each user
    signatures[:, j] = np.array([
        perm[movies].min() # mishash value for this user and this permutation
        for movies in user_movie_lists
    ])

print(signatures[0:5, :10])  # print first 5 users first 10 signature values
print(signatures.shape)



banded_signatures = signatures.reshape(n_users, bands, rows)
print(banded_signatures.shape)
print("User 0, band 0:", banded_signatures[0, 0, :])
print("User 0, band 1:", banded_signatures[0, 1, :])

import numpy as np

# We choose a random hash multiplier per row (for stable hashing)
# This mixes the row values into a single integer without collisions being catastrophic.
multipliers = np.random.randint(1, 2**31 - 1, size=rows, dtype=np.int64)

band_buckets = []   # list of: list of buckets, each bucket is a numpy array of user IDs

for b in range(bands):

    # Shape (n_users, rows)
    band = banded_signatures[:, b, :]

    # Compute hash per user:
    #   H(user) = sum(band[row] * multiplier[row])
    band_hashes = (band.astype(np.int64) * multipliers).sum(axis=1)

    # Sort users by hash so equal-hash users become adjacent
    order = np.argsort(band_hashes)
    sorted_hashes = band_hashes[order]

    # Find boundaries where the hash value changes
    changes = np.where(sorted_hashes[1:] != sorted_hashes[:-1])[0] + 1

    # Split into groups (buckets)
    groups = np.split(order, changes)

    # Keep buckets only if they contain at least 2 users
    buckets = [g for g in groups if len(g) >= 2]

    band_buckets.append(buckets)

print("Example buckets in band 0:", band_buckets[0][:5])


candidate_pairs = set()

for buckets in band_buckets:
    for bucket in buckets:
        bucket = np.asarray(bucket)
        bucket_size = bucket.size
       
        # Generate all unordered pairs inside the bucket
        i = np.repeat(np.arange(bucket_size - 1), np.arange(bucket_size - 1, 0, -1))
        j = np.concatenate([np.arange(x + 1, bucket_size) for x in range(bucket_size - 1)])
        pairs = np.stack((bucket[i], bucket[j]), axis=1)

        for u, v in pairs:
            if u < v:
                candidate_pairs.add((u, v))

candidate_pairs = np.array(list(candidate_pairs), dtype=np.int32)
print("Number of candidate pairs:", candidate_pairs.shape[0])


    
THRESHOLD = 0.5  # Jaccard similarity threshold

def jaccard(u_items, v_items):
    # intersection size
    inter = np.intersect1d(u_items, v_items, assume_unique=True).size
    # union size
    union = u_items.size + v_items.size - inter
    return inter / union if union > 0 else 0.0

# create or clear output file
open("similar_users.txt", "w").close()

for u, v in candidate_pairs:
    sim = jaccard(user_movie_lists[u], user_movie_lists[v])
    if sim > THRESHOLD:
        with open("similar_users.txt", "a") as f:
            # write u1,u2 to file
            f.write(f"{u},{v}\n")
