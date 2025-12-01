from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.dimension = DIMENSION
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function only loads one row in memory
        try:
            offset = int(row_num) * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"Error retrieving row {row_num}: {e}")
            return np.zeros(DIMENSION, dtype=np.float32)

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """Retrieve top-k nearest neighbors using IVF-FLAT index"""
        # Load index from disk (no caching allowed)
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        centroids = index_data['centroids']
        cluster_assignments = index_data['clusters']
        n_clusters = len(centroids)
        
        # Optimized nprobe values based on target performance
        if n_clusters >= 4000:
            nprobe = 10  # Reduced for speed while maintaining accuracy
        elif n_clusters >= 3000:
            nprobe = 8
        elif n_clusters >= 1000:
            nprobe = 5
        else:
            nprobe = max(2, n_clusters // 20)
        
        # Find nearest centroids using cosine similarity (vectorized)
        query_flat = query.flatten()
        query_norm = np.linalg.norm(query_flat)
        centroid_norms = np.linalg.norm(centroids, axis=1)
        centroid_scores = np.dot(centroids, query_flat) / (centroid_norms * query_norm + 1e-8)
        
        # Get top nprobe clusters
        top_cluster_ids = np.argpartition(centroid_scores, -nprobe)[-nprobe:]
        
        # Collect candidate indices from selected clusters
        candidates = []
        for cluster_id in top_cluster_ids:
            candidates.extend(cluster_assignments[cluster_id])
        
        # Quick path for small candidate sets
        num_candidates = len(candidates)
        if num_candidates <= top_k:
            return [int(c) for c in candidates[:top_k]]
        
        # Convert candidates to numpy array for efficient processing
        candidates_arr = np.array(candidates, dtype=np.int32)
        
        # Load candidate vectors using optimized batch loading
        candidate_vectors = np.zeros((num_candidates, DIMENSION), dtype=np.float32)
        
        # Open memmap once
        total_records = self._get_num_records()
        mmap_db = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                           shape=(total_records, DIMENSION))
        
        # Load in sorted order for better disk I/O
        sorted_indices = np.argsort(candidates_arr)
        sorted_candidates = candidates_arr[sorted_indices]
        
        # Batch load
        for i, idx in enumerate(sorted_candidates):
            candidate_vectors[sorted_indices[i]] = mmap_db[idx]
        
        del mmap_db  # Explicitly free memory
        
        # Vectorized cosine similarity computation
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        dot_products = np.dot(candidate_vectors, query_flat)
        scores = dot_products / (candidate_norms * query_norm + 1e-8)
        
        # Get top-k indices
        top_k_local_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_k_local_indices = top_k_local_indices[np.argsort(scores[top_k_local_indices])[::-1]]
        
        # Map back to global indices
        result_indices = [int(candidates[i]) for i in top_k_local_indices]
        
        return result_indices
    
    def _brute_force_retrieve(self, query, top_k):
        """Fallback brute force search"""
        scores = []
        num_records = self._get_num_records()
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2 + 1e-8)
        return cosine_similarity

    def _build_index(self):
        """Build IVF-FLAT index using K-means clustering"""
        print("Building IVF-FLAT index...")
        
        num_records = self._get_num_records()
        
        # Determine optimal cluster count based on database size
        if num_records >= 15_000_000:
            n_clusters = 4472  # sqrt(20M)
        elif num_records >= 8_000_000:
            n_clusters = 3162  # sqrt(10M)
        elif num_records >= 500_000:
            n_clusters = 1000
        else:
            n_clusters = max(100, int(np.sqrt(num_records)))
        
        print(f"Using {n_clusters} clusters for {num_records} records")
        
        # Sample vectors for clustering (memory efficient)
        sample_size = min(500_000, num_records)
        np.random.seed(DB_SEED_NUMBER)
        sample_indices = np.random.choice(num_records, sample_size, replace=False)
        
        # Load sample vectors in batches
        sample_vectors = np.zeros((sample_size, DIMENSION), dtype=np.float32)
        for i, idx in enumerate(sample_indices):
            sample_vectors[i] = self.get_one_row(int(idx))
            if i % 50000 == 0:
                print(f"  Loaded {i}/{sample_size} sample vectors")
        
        # Train K-means on samples
        print(f"Training K-means with {n_clusters} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=DB_SEED_NUMBER,
            batch_size=2048,
            max_iter=100,
            verbose=0
        )
        kmeans.fit(sample_vectors)
        centroids = kmeans.cluster_centers_
        
        print("Assigning all vectors to clusters...")
        # Assign all vectors to clusters in batches
        cluster_assignments = [[] for _ in range(n_clusters)]
        batch_size = 50000
        
        centroids_normalized = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        
        for start_idx in range(0, num_records, batch_size):
            end_idx = min(start_idx + batch_size, num_records)
            batch_vectors = np.zeros((end_idx - start_idx, DIMENSION), dtype=np.float32)
            
            for i, idx in enumerate(range(start_idx, end_idx)):
                batch_vectors[i] = self.get_one_row(idx)
            
            # Normalize and compute cosine similarity
            batch_normalized = batch_vectors / (np.linalg.norm(batch_vectors, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(batch_normalized, centroids_normalized.T)
            labels = np.argmax(similarities, axis=1)
            
            for i, label in enumerate(labels):
                cluster_assignments[label].append(start_idx + i)
            
            if start_idx % 500000 == 0:
                print(f"  Processed {start_idx}/{num_records} vectors")
        
        # Save index to disk
        self._save_index(centroids, cluster_assignments)
        print(f"Index built successfully!")
    
    def _save_index(self, centroids, clusters):
        """Save index to disk as compressed pickle"""
        # Convert lists to numpy arrays for better compression
        compressed_clusters = [np.array(cluster, dtype=np.int32) for cluster in clusters]
        
        index_data = {
            'centroids': centroids.astype(np.float32),
            'clusters': compressed_clusters,
            'n_clusters': len(centroids),
            'dimension': DIMENSION
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024)
        print(f"Index saved to {self.index_path} ({index_size_mb:.2f} MB)")
    
    def _load_index(self):
        """Load index from disk"""
        if not os.path.exists(self.index_path):
            return None
        
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        return index_data


