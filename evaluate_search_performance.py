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

print("✅ Connected to Pinecone index")
print(f"📊 Index stats: {index.describe_index_stats()}")


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
# TEST QUERIES WITH GROUND TRUTH
# ============================================

# Define your test queries with known good answers
TEST_QUERIES = [
    {
        "query": "Find the best science fiction movies with high ratings",
        "min_rating": 4.0,
        "description": "Looking for highly-rated sci-fi films",
        
        # List of titles you expect to see (for hit rate and RR)
        "relevant_titles": [
            "The Matrix",
            "Inception", 
            "Interstellar",
            "Blade Runner",
            "2001: A Space Odyssey"
        ],
        
        # How relevant is each movie? (for NDCG)
        # 5 = perfect match, 0 = not relevant
        "relevance_scores": {
            "The Matrix": 5,
            "Inception": 5,
            "Interstellar": 5,
            "Blade Runner": 4,
            "2001: A Space Odyssey": 5,
            "Star Wars": 4,
            "Alien": 4
        }
    },
    
    {
        "query": "Show me classic horror movies",
        "min_rating": 0.0,
        "description": "Looking for well-known horror films",
        
        "relevant_titles": [
            "The Exorcist",
            "Psycho",
            "The Shining",
            "Halloween",
            "A Nightmare on Elm Street"
        ],
        
        "relevance_scores": {
            "The Exorcist": 5,
            "Psycho": 5,
            "The Shining": 5,
            "Halloween": 4,
            "A Nightmare on Elm Street": 4,
            "The Conjuring": 3,
            "Scream": 3
        }
    },
    
    {
        "query": "Best romantic comedies",
        "min_rating": 3.5,
        "description": "Looking for feel-good rom-coms",
        
        "relevant_titles": [
            "When Harry Met Sally",
            "Notting Hill",
            "The Proposal",
            "Crazy, Stupid, Love"
        ],
        
        "relevance_scores": {
            "When Harry Met Sally": 5,
            "Notting Hill": 5,
            "The Proposal": 4,
            "Crazy, Stupid, Love": 4,
            "10 Things I Hate About You": 4
        }
    }
]


# ============================================
# RUN EVALUATION
# ============================================

def evaluate_query(query_data, k=10):
    """
    Evaluate a single query and explain the results
    """
    print("\n" + "="*80)
    print(f"📝 QUERY: {query_data['query']}")
    print(f"💡 Looking for: {query_data['description']}")
    print("="*80)
    
    # Step 1: Search
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
    
    # Step 2: Show results
    print(f"\n🎬 Top {min(10, len(retrieved_titles))} Results:")
    print("-" * 80)
    for i, match in enumerate(results['matches'][:10], 1):
        title = match['metadata']['title']
        rating = match['metadata']['rating']
        score = match['score']
        
        # Check if relevant
        is_relevant = "✅" if title in relevant_titles else "❌"
        
        # Show relevance score if available
        rel_score = relevance_scores.get(title, 0)
        if rel_score > 0:
            print(f"  {i}. {is_relevant} {title} (⭐{rating:.1f}, 🎯{score:.3f}, Relevance: {rel_score}/5)")
        else:
            print(f"  {i}. {is_relevant} {title} (⭐{rating:.1f}, 🎯{score:.3f})")
    
    # Step 3: Calculate metrics
    print(f"\n📊 METRICS:")
    print("-" * 80)
    
    # Reciprocal Rank
    print("\n1️⃣ Reciprocal Rank (How fast did we find something good?)")
    rr = reciprocal_rank(retrieved_titles, relevant_titles)
    print(f"   RR = {rr:.3f}")
    if rr >= 0.5:
        print(f"   ✅ Great! Found relevant result in top {int(1/rr)} positions")
    elif rr > 0:
        print(f"   ⚠️  Relevant result not found until position {int(1/rr)}")
    else:
        print(f"   ❌ No relevant results found")
    
    # NDCG
    print(f"\n2️⃣ NDCG@{k} (How good is our ranking overall?)")
    ndcg = ndcg_at_k(retrieved_titles, relevance_scores, k)
    if ndcg >= 0.8:
        print(f"   ✅ Excellent ranking!")
    elif ndcg >= 0.6:
        print(f"   👍 Good ranking")
    elif ndcg >= 0.4:
        print(f"   ⚠️  Fair ranking - could be better")
    else:
        print(f"   ❌ Poor ranking - needs improvement")
    
    # Hit Rate
    print(f"\n3️⃣ Hit Rate@{k} (Did we find ANY relevant results?)")
    hit = hit_rate_at_k(retrieved_titles, relevant_titles, k)
    print(f"   Hit@{k} = {hit:.1f}")
    if hit == 1.0:
        print(f"   ✅ Success! Found at least one relevant movie")
    else:
        print(f"   ❌ Failed to find any relevant movies")
    
    return {
        "query": query_data['query'],
        "reciprocal_rank": rr,
        f"ndcg@{k}": ndcg,
        f"hit@{k}": hit
    }


# ============================================
# EVALUATE ALL QUERIES
# ============================================

print("\n" + "🎬" * 40)
print("SEMANTIC SEARCH EVALUATION")
print("🎬" * 40)

all_results = []
for query_data in TEST_QUERIES:
    result = evaluate_query(query_data, k=10)
    all_results.append(result)


# ============================================
# AGGREGATE RESULTS
# ============================================

print("\n" + "="*80)
print("📈 OVERALL PERFORMANCE")
print("="*80)

avg_rr = np.mean([r['reciprocal_rank'] for r in all_results])
avg_ndcg = np.mean([r['ndcg@10'] for r in all_results])
avg_hit = np.mean([r['hit@10'] for r in all_results])

print(f"\n📊 Average Metrics Across All Queries:")
print(f"   Mean Reciprocal Rank: {avg_rr:.3f}")
print(f"   Mean NDCG@10: {avg_ndcg:.3f}")
print(f"   Mean Hit Rate@10: {avg_hit:.3f}")

print(f"\n🎯 Interpretation:")
if avg_ndcg >= 0.7 and avg_rr >= 0.5:
    print("   ✅ Your search is performing well!")
    print("   → Users are finding relevant movies quickly")
    print("   → Rankings are high quality")
elif avg_ndcg >= 0.5 or avg_rr >= 0.3:
    print("   ⚠️  Your search is decent but has room for improvement")
    print("   → Consider adding more movies to your index")
    print("   → Check if your embedding model is appropriate")
else:
    print("   ❌ Your search needs improvement")
    print("   → Review your test queries - are they realistic?")
    print("   → Check if your index has relevant movies")
    print("   → Consider using a different embedding model")

print("\n" + "="*80)
print("✅ Evaluation Complete!")
print("="*80)


# ============================================
# TIPS FOR IMPROVING YOUR SEARCH
# ============================================

print("""
💡 TIPS FOR BETTER SEARCH PERFORMANCE:

1. ADD MORE MOVIES
   - More data = better results
   - Aim for at least 10,000+ movies

2. IMPROVE EMBEDDINGS
   - Current: text-embedding-3-small (512 dim)
   - Better: text-embedding-3-large (3072 dim)
   - Trade-off: cost vs quality

3. ENRICH METADATA
   - Add genres, actors, directors
   - Include plot summaries
   - More text = better semantic understanding

4. TUNE YOUR TEST SET
   - Use realistic queries (what users actually ask)
   - Include diverse query types
   - Add more relevance judgments

5. EXPERIMENT WITH FILTERS
   - Try different min_rating thresholds
   - Filter by year, genre, etc.
   - Balance between quantity and quality
""")