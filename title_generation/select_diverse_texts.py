import argparse
import json
import numpy as np
import random
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select diverse text samples from a dataset. It uses k-means clustering on text embeddings to select diverse samples. "
                    "The input is a directory containing JSONL files with text chunks and .npy files containing embeddings. "
                    "The JSONL records contain fields 'text', 'language', and 'vector_index'."
                    "The script randomly read JSONL files from the input directory until a specified number of samples is collected, "
                    "loads the corresponding embeddings, performs k-means clustering, and selects one sample from each cluster to ensure diversity."
                    "The selected text samples are copied to the specified output JSONL file.")
    parser.add_argument("--input-dir", required=True, help="Path to the input directory containing JSONL and .npy files.")
    parser.add_argument("--output-file", required=True, help="Path to save the selected diverse texts.")
    parser.add_argument("--load-limit", type=int, default=100000, help="Maximum number of samples to load for diversity selection.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of diverse samples to select.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--pca-dimensions", type=int, help="Number of PCA dimensions to reduce embeddings to before clustering. Default is no PCA.")
    parser.add_argument('--languages', nargs='*', help="Optional list of languages to filter texts by (e.g., 'en', 'de'). If not specified, all languages are included.")
    parser.add_argument('--record-subsampling', type=int, default=10, help="Subsample records by this factor when loading (e.g., 2 means take every second record). Default is 1 (no subsampling).")
    parser.add_argument('--samples-per-cluster', type=int, default=1, help="Number of samples to select per cluster. Default is 1.")
    return parser.parse_args()

def load_samples_randomly(input_dir: Path, load_limit: int, random_seed: int, languages: List[str] = None,
                          subsample_factor: int = 1) -> Tuple[List[Dict], np.ndarray]:
    """
    Randomly read JSONL files and load corresponding embeddings until load_limit is reached.

    Returns:
        Tuple of (list of text records, numpy array of embeddings)
    """
    random.seed(random_seed)

    # Find all JSONL files in the directory
    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {input_dir}")

    # Shuffle the files to read them randomly
    random.shuffle(jsonl_files)

    all_records = []
    all_embeddings = []

    logging.info(f"Found {len(jsonl_files)} JSONL files in {input_dir}")
    logging.info(f"Loading up to {load_limit} samples...")

    for jsonl_file in jsonl_files:
        if len(all_records) >= load_limit:
            break

        # Replace .jsonl with _embeddings.npy in the filename
        npy_file = jsonl_file.with_name(jsonl_file.stem + "_embeddings.npy")

        if not npy_file.exists():
            logging.warning(f"No .npy file found for {jsonl_file}, skipping...")
            continue

        try:
            # Load JSONL records
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                records = [l.strip() for l in f]
                records = [line for line in records if line]
                all_records_count = len(records)
                records = records[::subsample_factor]
            records = [json.loads(line) for line in records]
            record_count = len(records)
            if languages is not None:
                records = [record for record in records if record.get('language') in languages]
            filtered_count = len(records)

            if records:
                # Load embeddings
                embeddings = np.load(npy_file)

                # Verify that the number of records matches embeddings
                if all_records_count != len(embeddings):
                    logging.warning(f"Mismatch in {jsonl_file}: {len(records)} records vs {len(embeddings)} embeddings, skipping...")
                    continue

                embedding_indices = [record['vector_index'] for record in records]
                embeddings = embeddings[embedding_indices]

                all_records.extend(records)
                all_embeddings.append(embeddings)

            logging.info(f"chunks: {record_count}, filtered: {filtered_count}, total loaded: {len(all_records)}, file {jsonl_file}")

        except Exception as e:
            logging.error(f"Error loading {jsonl_file}: {e}, skipping...")
            continue

    if not all_records:
        raise ValueError("No samples were loaded successfully")

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)

    logging.info(f"Successfully loaded {len(all_records)} samples with embeddings of shape {all_embeddings.shape}")

    return all_records, all_embeddings


def apply_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Apply PCA to reduce dimensionality of embeddings."""
    print(f"Applying PCA to reduce from {embeddings.shape[1]} to {n_components} dimensions...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA completed. Explained variance: {explained_variance:.4f}")
    return reduced_embeddings


def select_diverse_samples(
        records: List[Dict], embeddings: np.ndarray, num_samples: int, random_seed: int,
        samples_per_cluster: int = 1
        ) -> List[Dict]:
    """
    Use k-means clustering to select diverse samples.
    Selects one sample from each cluster (closest to centroid).
    """
    if num_samples > len(records):
        print(f"Warning: Requested {num_samples} samples but only {len(records)} available. Using all samples.")
        return records

    logging.info(f"Performing k-means clustering with k={num_samples}...")

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_samples, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Select one sample from each cluster (closest to centroid)
    selected_indices = []

    for cluster_id in range(num_samples):
        # Find all samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            logging.warning(f"Warning: Cluster {cluster_id} is empty")
            continue

        if samples_per_cluster > 1:
            if len(cluster_indices) <= samples_per_cluster:
                selected_indices.extend(cluster_indices.tolist())
            else:
                selected = random.sample(cluster_indices.tolist(), samples_per_cluster)
                selected_indices.extend(selected)
        else:
            # Find the sample closest to the centroid
            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx_in_cluster = np.argmin(distances)
            closest_idx = cluster_indices[closest_idx_in_cluster]
            selected_indices.append(closest_idx)

    logging.info(f"Selected {len(selected_indices)} diverse samples from {num_samples} clusters")

    # Return the selected records
    selected_records = [records[idx] for idx in selected_indices]

    return selected_records


def save_results(records: List[Dict], output_file: Path):
    """Save selected records to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(records)} diverse samples to {output_file}")


def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    # Load samples randomly
    records, embeddings = load_samples_randomly(input_dir, args.load_limit, args.random_seed, args.languages)

    # Apply PCA if requested
    if args.pca_dimensions:
        if args.pca_dimensions >= embeddings.shape[1]:
            print(f"Warning: PCA dimensions ({args.pca_dimensions}) >= embedding dimensions ({embeddings.shape[1]}), skipping PCA")
        else:
            embeddings = apply_pca(embeddings, args.pca_dimensions)

    # Select diverse samples using k-means
    selected_records = select_diverse_samples(
        records, embeddings, args.num_samples, args.random_seed, args.samples_per_cluster)

    # Save results
    save_results(selected_records, output_file)

    print("Done!")


if __name__ == "__main__":
    main()
