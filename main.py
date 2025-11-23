#!/usr/bin/env python3
"""
LSH-based User Similarity Finder for Netflix Data
Finds pairs of users with Jaccard similarity > threshold using Locality Sensitive Hashing
"""
import argparse
import sys
import os
import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict


def load_data(file_path):
    """Load and preprocess Netflix rating data into CSR sparse matrix."""
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    data = np.load(file_path, mmap_mode='r')
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Expected a 2D array with 3 columns: user_id, movie_id, rating")

    users = data[:, 0].astype(int)
    movies = data[:, 1].astype(int)

    n_users = users.max()
    n_movies = movies.max()

    print(f"  Loaded {data.shape[0]} interactions")
    print(f"  Users: {n_users}, Movies: {n_movies}")

    # Convert to 0-indexed
    users -= 1
    movies -= 1

    # Create sparse matrix (ratings irrelevant, just presence)
    data_values = np.ones_like(users, dtype=np.uint8)
    coo = coo_matrix((data_values, (users, movies)), shape=(n_users, n_movies), dtype=bool)
    csr = coo.tocsr()

    print(f"  Created sparse matrix: shape={csr.shape}, nnz={csr.nnz}")

    # Create user movie lists for efficient minhashing
    user_movie_lists = [
        csr.indices[csr.indptr[i]:csr.indptr[i+1]]
        for i in range(n_users)
    ]

    return csr, user_movie_lists, n_users, n_movies


def generate_signatures(user_movie_lists, n_users, n_movies, k, seed):
    """Generate MinHash signatures for all users."""
    print(f"Generating signatures with k={k} hash functions...")

    np.random.seed(seed)

    # Create k random permutations of movie indices
    permutations = np.array([
        np.random.permutation(n_movies)
        for _ in range(k)
    ], dtype=np.int32)

    signatures = np.empty(shape=(n_users, k), dtype=np.int32)

    # Compute minhash signatures
    for j in range(k):
        if (j + 1) % 20 == 0:
            print(f"  Progress: {j+1}/{k} hash functions")

        perm = permutations[j]
        for i in range(n_users):
            movies = user_movie_lists[i]
            signatures[i, j] = perm[movies].min()

    print(f"  Signatures generated: shape={signatures.shape}")
    return signatures


def find_candidate_pairs(signatures, bands, rows):
    """Use banding technique to find candidate pairs."""
    print(f"Finding candidate pairs using {bands} bands with {rows} rows each...")

    n_users = signatures.shape[0]
    banded_signatures = signatures.reshape(n_users, bands, rows)

    # Buckets: {band_id: {hash: [user_ids]}}
    buckets = defaultdict(lambda: defaultdict(list))

    print(f"  Hashing users into buckets...")
    for band_id in range(bands):
        if (band_id + 1) % 5 == 0 or band_id == 0:
            print(f"    Progress: {band_id+1}/{bands} bands processed")

        for user_id in range(n_users):
            band_signature = tuple(banded_signatures[user_id, band_id, :])
            bucket_hash = hash(band_signature)
            buckets[band_id][bucket_hash].append(user_id)

    # Extract candidate pairs
    print(f"  Extracting candidate pairs from buckets...")
    candidate_pairs = set()
    for band_id in range(bands):
        if (band_id + 1) % 5 == 0 or band_id == 0:
            print(f"    Progress: {band_id+1}/{bands} bands, {len(candidate_pairs)} candidates so far")

        for bucket_hash, user_list in buckets[band_id].items():
            if len(user_list) > 1:
                # Generate all pairs in this bucket
                for i in range(len(user_list)):
                    for j in range(i + 1, len(user_list)):
                        u1, u2 = user_list[i], user_list[j]
                        if u1 > u2:
                            u1, u2 = u2, u1
                        candidate_pairs.add((u1, u2))

    print(f"  Found {len(candidate_pairs)} candidate pairs")
    return candidate_pairs


def jaccard_similarity(movies1, movies2):
    """Compute Jaccard similarity between two sets of movies."""
    intersection = np.intersect1d(movies1, movies2, assume_unique=True)
    union = np.union1d(movies1, movies2)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def filter_similar_pairs(candidate_pairs, user_movie_lists, threshold):
    """Filter candidate pairs by Jaccard similarity threshold."""
    print(f"Filtering pairs with similarity > {threshold}...")

    similar_pairs = []
    total = len(candidate_pairs)

    for idx, (u1, u2) in enumerate(candidate_pairs):
        if (idx + 1) % 100000 == 0:
            print(f"  Progress: {idx+1}/{total} pairs checked")

        similarity = jaccard_similarity(user_movie_lists[u1], user_movie_lists[u2])
        if similarity > threshold:
            similar_pairs.append((u1, u2, similarity))

    print(f"  Found {len(similar_pairs)} similar pairs")
    return similar_pairs


def write_output(similar_pairs, output_file):
    """Write similar pairs to output file."""
    print(f"Writing output to {output_file}...")

    # Convert back to 1-indexed user IDs
    with open(output_file, 'w') as f:
        for u1, u2, _ in similar_pairs:
            f.write(f"{u1+1},{u2+1}\n")

    print(f"  Wrote {len(similar_pairs)} pairs")


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
        default=10,
        help='Number of bands (default: 10, giving 12 rows per band)'
    )

    args = parser.parse_args()

    # Validate parameters
    if args.k % args.bands != 0:
        print(f"Error: k ({args.k}) must be divisible by bands ({args.bands})")
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1:
        print(f"Error: threshold must be between 0 and 1")
        sys.exit(1)

    rows = args.k // args.bands

    print("=" * 60)
    print("LSH User Similarity Finder")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Random seed: {args.seed}")
    print(f"Parameters: k={args.k}, bands={args.bands}, rows={rows}")
    print(f"Similarity threshold: {args.threshold}")
    print("=" * 60)

    # Execute pipeline
    csr, user_movie_lists, n_users, n_movies = load_data(args.input)
    signatures = generate_signatures(user_movie_lists, n_users, n_movies, args.k, args.seed)
    candidate_pairs = find_candidate_pairs(signatures, args.bands, rows)
    similar_pairs = filter_similar_pairs(candidate_pairs, user_movie_lists, args.threshold)
    write_output(similar_pairs, args.output)

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
