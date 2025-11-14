"""
Author: Consuelo Cuevas
Date: 2025
"""

import os
import time
import logging
import openai
import tqdm
from pinecone import Pinecone, ServerlessSpec
import pinecone_datasets
from dotenv import load_dotenv

# ============================================
# CONFIGURATION
# ============================================

# Load environment variables from .env file
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENC_MODEL = "text-embedding-3-small"  # OpenAI embedding model
EMB_DIM = 512  # Embedding dimensions (512 for efficiency, 1536 for max quality)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = 'semantic-search-movie-demo'  # Name of your Pinecone index
CLOUD_PROVIDER = 'aws'  # Cloud provider (aws, gcp, or azure)
REGION = 'us-east-1'  # Region for serverless deployment

# Processing configuration
BATCH_SIZE = 100  # Number of vectors to upsert at once
MAX_MOVIES = 1000  # Limit dataset to first N movies (None for all ~1M movies)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Validate API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ============================================
# HELPER FUNCTIONS
# ============================================


def get_embedding(text_to_embed):
    """
    Generate an embedding vector for the given text.

    Args:
        text_to_embed (str): Text to convert into an embedding vector

    Returns:
        list: Embedding vector (list of floats)
    """
    try:
        response = openai.embeddings.create(
            model=ENC_MODEL,
            input=[text_to_embed],
            dimensions=EMB_DIM
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise


def create_pinecone_index(pc, index_name, dimension, metric='cosine'):
    """
    Create a new Pinecone index if it doesn't already exist.

    Args:
        pc: Pinecone client instance
        index_name (str): Name of the index to create
        dimension (int): Dimension of the vectors
        metric (str): Distance metric to use ('cosine', 'euclidean', or 'dotproduct')

    Returns:
        None
    """
    # Check if index already exists
    existing_indexes = [p['name'] for p in pc.list_indexes()]

    if index_name in existing_indexes:
        logging.info(f"Index '{index_name}' already exists. Loading index.")
        return

    logging.info(f"Creating new index '{index_name}'...")

    # Configure serverless spec
    spec = ServerlessSpec(cloud=CLOUD_PROVIDER, region=REGION)

    # Create index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=spec
    )

    logging.info(f"Index '{index_name}' created successfully!")

    # Wait for index to be ready
    logging.info("Waiting for index to be ready...")
    time.sleep(5)


def load_dataset(dataset_name='movielens-user-ratings', max_rows=None):
    """
    Load the MovieLens dataset from Pinecone's public datasets.

    Args:
        dataset_name (str): Name of the dataset to load
        max_rows (int, optional): Limit dataset to first N rows. None for all rows.

    Returns:
        Dataset: Pinecone dataset object
    """
    logging.info(f"Loading dataset '{dataset_name}'...")
    dataset = pinecone_datasets.load_dataset(dataset_name)

    original_size = len(dataset.documents)
    logging.info(f"Dataset loaded with {original_size:,} movies")

    # Limit dataset size if specified
    if max_rows and max_rows < original_size:
        logging.info(f"Limiting dataset to first {max_rows:,} movies...")
        dataset.documents.drop(dataset.documents.index[max_rows:], inplace=True)
        logging.info(f"Dataset reduced to {len(dataset.documents):,} movies")

    return dataset


def upsert_embeddings(index, dataset, batch_size=100):
    """
    Generate embeddings for all movies and upload them to Pinecone.

    Args:
        index: Pinecone index object
        dataset: Pinecone dataset containing movie data
        batch_size (int): Number of vectors to upload per batch

    Returns:
        int: Total number of vectors uploaded
    """
    batch = []
    total_rows = len(dataset.documents)
    vectors_uploaded = 0

    logging.info(f"Starting to process {total_rows:,} movies...")
    logging.info("This may take a while depending on dataset size and API rate limits.")

    with tqdm.tqdm(total=total_rows, desc="Processing movies") as pbar:
        for idx, (_, row) in enumerate(dataset.documents.iterrows()):
            try:
                # Extract movie information
                vector_id = str(row['id'])
                title = row['blob']['title']
                rating = row['blob']['rating']

                # Create a text description for embedding
                # Format: "[Title] Movie Name [With Rating] X.X"
                movie_info = f"[Title] {title} [With Rating] {rating}"

                # Generate embedding
                embedding_vector = get_embedding(movie_info)

                # Prepare metadata (stored alongside the vector)
                metadata = {
                    'title': title,
                    'rating': rating
                }

                # Add to batch
                batch.append({
                    "id": vector_id,
                    "values": embedding_vector,
                    "metadata": metadata
                })

                # Upload batch when it reaches batch_size
                if len(batch) == batch_size:
                    logging.info(f"Uploading batch ending at index {idx} ({vectors_uploaded + batch_size:,} total)")
                    index.upsert(vectors=batch)
                    vectors_uploaded += len(batch)
                    batch = []

                    # Small delay to avoid rate limiting
                    time.sleep(0.5)

                pbar.update(1)

            except Exception as e:
                logging.error(f"Error processing movie at index {idx}: {e}")
                logging.error(f"Movie data: {row['blob']}")
                continue

        # Upload remaining vectors
        if batch:
            logging.info(f"Uploading final batch ({len(batch)} vectors)")
            index.upsert(vectors=batch)
            vectors_uploaded += len(batch)

    logging.info(f"Finished! Uploaded {vectors_uploaded:,} vectors to Pinecone.")
    return vectors_uploaded


def test_index(index, test_queries=None, top_k=5):
    """
    Test the Pinecone index with sample queries.

    Args:
        index: Pinecone index object
        test_queries (list, optional): List of test query strings
        top_k (int): top closest entries to return

    Returns:
        None
    """
    if test_queries is None:
        test_queries = [
            "What was the movie titled Whiplash's rating?",
            "Find the best action movies from the 1990s",
            "Show me highly rated sci-fi films"
        ]

    logging.info("\n" + "="*80)
    logging.info("TESTING INDEX")
    logging.info("="*80)

    for query in test_queries:
        logging.info(f"\nQuery: {query}")
        logging.info("-" * 80)

        try:
            # Generate query embedding
            query_embedding = get_embedding(query)

            # Search without filters
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            logging.info(f"Top {len(results['matches'])} results:")
            for i, match in enumerate(results['matches'], 1):
                title = match['metadata']['title']
                rating = match['metadata']['rating']
                score = match['score']
                logging.info(f"  {i+1}. {title} (Rating: {rating:.1f}, Similarity: {score:.3f})")

            # Search with rating filter (>= 4.0)
            logging.info("\nFiltered results (rating >= 4.0):")
            filtered_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"rating": {"$gte": 4.0}}
            )

            for i, match in enumerate(filtered_results['matches'], 1):
                title = match['metadata']['title']
                rating = match['metadata']['rating']
                score = match['score']
                logging.info(f"  {i}. {title} (Rating: {rating:.1f}, Similarity: {score:.3f})")

            print("")
        except Exception as e:
            logging.error(f"Error testing query '{query}': {e}")

    logging.info("\n" + "="*80)
    logging.info("TESTING COMPLETE")
    logging.info("="*80)


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main function to orchestrate the entire setup process.
    """
    logging.info("Starting Pinecone index setup...")
    logging.info("Configuration:")
    logging.info(f"  - Index name: {INDEX_NAME}")
    logging.info(f"  - Embedding model: {ENC_MODEL}")
    logging.info(f"  - Embedding dimensions: {EMB_DIM}")
    logging.info(f"  - Batch size: {BATCH_SIZE}")
    logging.info(f"  - Max movies: {MAX_MOVIES if MAX_MOVIES else 'All'}")

    try:
        # Step 1: Initialize Pinecone client
        logging.info("\nStep 1: Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info("Connected to Pinecone successfully!")

        # Step 2: Create index if needed
        logging.info("\nStep 2: Setting up Pinecone index...")
        create_pinecone_index(pc, INDEX_NAME, EMB_DIM, metric='cosine')

        # Step 3: Connect to index
        logging.info("\nStep 3: Connecting to index...")
        index = pc.Index(INDEX_NAME)

        # Display current index stats
        stats = index.describe_index_stats()
        logging.info(f"Current index stats: {stats}")

        # Step 4: Load dataset
        logging.info("\nStep 4: Loading MovieLens dataset...")
        dataset = load_dataset('movielens-user-ratings', max_rows=MAX_MOVIES)

        # Display sample data
        logging.info("\nSample movie data:")
        sample_movie = dataset.documents.iloc[0]['blob']
        logging.info(f"  Title: {sample_movie['title']}")
        logging.info(f"  Rating: {sample_movie['rating']}")

        # Step 5: Generate and upload embeddings
        logging.info("\nStep 5: Generating embeddings and uploading to Pinecone...")

        # Ask for confirmation if processing many movies
        if not MAX_MOVIES or MAX_MOVIES > 1000:
            response = input("\nYou're about to process many movies. Continue? (yes/no): ")
            if response.lower() != 'yes':
                logging.info("Aborted by user.")
                return

        _ = upsert_embeddings(index, dataset, batch_size=BATCH_SIZE)

        # Step 6: Verify upload
        logging.info("\nStep 6: Verifying upload...")
        time.sleep(2)  # Wait for index to update
        final_stats = index.describe_index_stats()
        logging.info(f"Final index stats: {final_stats}")

        # Step 7: Test the index
        logging.info("\nStep 7: Testing index with sample queries...")
        test_index(index)

        # Success message
        logging.info("\n" + "="*80)
        logging.info("SETUP COMPLETE!")
        logging.info("="*80)
        logging.info(f"Your Pinecone index '{INDEX_NAME}' is ready to use!")
        logging.info(f"Total vectors in index: {final_stats['total_vector_count']:,}")
        logging.info("\nYou can now use this index in your Streamlit app.")
        logging.info("Make sure to use the same index name in your app configuration.")

    except Exception as e:
        logging.error(f"\nError during setup: {e}")
        logging.error("Setup failed. Please check the error message above and try again.")
        raise


if __name__ == "__main__":
    main()
