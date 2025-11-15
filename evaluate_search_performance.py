import os
import numpy as np
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from setup_pinecone_index import get_embedding

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "semantic-search-movie-demo"
ENC_MODEL = "text-embedding-3-small"
EMB_DIM = 512

# Initialize APIs
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("Connected to Pinecone index")
print(f"Index stats:\n{index.describe_index_stats()}")


def reciprocal_rank(retrieved_titles, relevant_titles):
    """
    Reciprocal Rank (RR): How quickly do we find the first relevant result?

    Formula: RR = 1 / (position of first relevant item)

    Interpretation:
        - 1.0 = Perfect! First result is relevant
        - 0.5 = First relevant at position 2
        - 0.1 = First relevant at position 10
        - 0.0 = No relevant results found
    """
    for position, title in enumerate(retrieved_titles, start=1):
        if title in relevant_titles:
            rr = 1.0 / position
            print(f"First relevant result at position {position}")
            return rr
    print("No relevant results found!")
    return 0.0


def ndcg_at_k(retrieved_titles, relevance_scores, k):
    """
    NDCG@K: How good is our ranking, considering BOTH relevance AND position?

    Graded Relevance of results:
       - 5 = Highly relevant (perfect match)
       - 4 = Very relevant
       - 3 = Moderately relevant
       - 2 = Slightly relevant
       - 1 = Barely relevant
       - 0 = Not relevant

    Formula:
        DCG = Σ (2^relevance - 1) / log2(position + 1)
        NDCG = DCG / IDCG (where IDCG is the ideal DCG)

    Example:
        Your ranking:
        1. "Inception" (relevance: 5)
        2. "Some Bad Movie" (relevance: 0)
        3. "The Matrix" (relevance: 5)

        NDCG ≈ 0.85 (good but not perfect, because "The Matrix" should be higher)

        Perfect ranking would be:
        1. "Inception" (5)
        2. "The Matrix" (5)
        3. "Some Bad Movie" (0)

        NDCG = 1.0 (perfect!)

    Interpretation:
        - 1.0 = Perfect ranking
        - 0.8-1.0 = Excellent
        - 0.6-0.8 = Good
        - 0.4-0.6 = Fair
        - <0.4 = Needs improvement

    """
    def dcg_at_k(scores, k):
        """Calculate Discounted Cumulative Gain"""
        scores = scores[:k]
        if len(scores) == 0:
            return 0.0
        return sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(scores))

    # Get relevance scores for retrieved items (0 if not in relevance_scores)
    retrieved_k = retrieved_titles[:k]
    retrieved_scores = [relevance_scores.get(title, 0.0) for title in retrieved_k]

    # Calculate DCG (actual ranking)
    dcg = dcg_at_k(retrieved_scores, k)

    # Calculate IDCG (ideal/perfect ranking)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg

    # Show breakdown
    print(f"DCG (your ranking): {dcg:.2f}")
    print(f"IDCG (perfect ranking): {idcg:.2f}")
    print(f"NDCG@{k}: {ndcg:.3f}")

    return ndcg


def hit_rate_at_k(retrieved_titles, relevant_titles, k):
    """
    Hit Rate@K: Did we find AT LEAST ONE relevant result in top-K?

    Simple yes/no metric:
        - 1.0 = Found at least one relevant result
        - 0.0 = Found zero relevant results

    Example:
        Retrieved top 5: ["Bad", "Bad", "Good", "Bad", "Bad"]
        Relevant: ["Good", "Great"]

        Hit@5 = 1.0 (we found "Good")

    Why use it?
        Simple to understand and useful for checking if your system finds
        ANYTHING relevant. If hit rate is low, something is seriously wrong!
    """
    retrieved_k = retrieved_titles[:k]
    found = any(title in relevant_titles for title in retrieved_k)
    return 1.0 if found else 0.0


# ============================================
# RUN EVALUATION
# ============================================

def evaluate_query(query_data, k=10):
    """
    Evaluate a single query and explain the results
    """
    print("\n" + "="*80)
    print(f"QUERY: {query_data['query']}")
    print(f"Looking for: {query_data['description']}")
    print("="*80)

    # Search
    query_embedding = get_embedding(query_data['query'])
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
        filter={"rating": {"$gte": query_data.get('min_rating', 0.0)}}
    )

    # Extract titles
    retrieved_titles = [m['metadata']['title'] for m in results['matches']]
    relevant_titles = query_data['relevant_titles']
    relevance_scores = query_data['relevance_scores']

    # Show results
    print(f"\nTop {min(10, len(retrieved_titles))} Results:")
    print("-" * 80)
    for i, match in enumerate(results['matches'][:10], 1):
        title = match['metadata']['title']
        rating = match['metadata']['rating']
        score = match['score']

        # Check if relevant
        is_relevant = "✓" if title in relevant_titles else "✗"

        # Show relevance score if available
        rel_score = relevance_scores.get(title, 0)
        if rel_score > 0:
            print(f"  {i}. {is_relevant} {title} ({rating:.1f}, {score:.3f}, Relevance: {rel_score}/5)")
        else:
            print(f"  {i}. {is_relevant} {title} ({rating:.1f}, {score:.3f})")

    # Calculate metrics
    print("\nMETRICS:")
    print("-" * 80)

    # Reciprocal Rank
    print("\nReciprocal Rank (How fast did we find something good?)")
    rr = reciprocal_rank(retrieved_titles, relevant_titles)
    print(f"   RR = {rr:.3f}")
    if rr >= 0.5:
        print(f"   ✓ Great! Found relevant result in top {int(1/rr)} positions")
    elif rr > 0:
        print(f"   ⚠️  Relevant result not found until position {int(1/rr)}")
    else:
        print("   ✗ No relevant results found")

    # NDCG
    print(f"\nNDCG@{k} (How good is our ranking overall?)")
    ndcg = ndcg_at_k(retrieved_titles, relevance_scores, k)
    if ndcg >= 0.8:
        print("   ✓ Excellent ranking!")
    elif ndcg >= 0.6:
        print("   👍 Good ranking")
    elif ndcg >= 0.4:
        print("   ⚠️  Fair ranking - could be better")
    else:
        print("   ✗ Poor ranking - needs improvement")

    # Hit Rate
    print(f"\nHit Rate@{k} (Did we find ANY relevant results?)")
    hit = hit_rate_at_k(retrieved_titles, relevant_titles, k)
    print(f"   Hit@{k} = {hit:.1f}")
    if hit == 1.0:
        print("   ✓ Success! Found at least one relevant movie")
    else:
        print("   ✗ Failed to find any relevant movies")

    return {
        "query": query_data['query'],
        "reciprocal_rank": rr,
        f"ndcg@{k}": ndcg,
        f"hit@{k}": hit
    }

