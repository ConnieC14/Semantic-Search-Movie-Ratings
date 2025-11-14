import time
import openai
import tqdm
import numpy as np
from setup_pinecone_index import get_embedding

# ============================================
# METRIC FUNCTIONS
# ============================================


def is_fuzzy_match(test_title, retrieved_title):
    """
    Fuzzy matching: allows for matching similar words/phrases
    """
    return test_title.lower().strip() in retrieved_title.lower().strip()


def reciprocal_rank(retrieved_titles, relevant_titles):
    """How quickly do we find the first relevant result?"""
    for position, retrieved_title in enumerate(retrieved_titles, start=1):
        for test_title in relevant_titles:
            if is_fuzzy_match(test_title, retrieved_title):
                return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_titles, relevance_scores, k):
    """ Calculate Normalized Discounted Cumulative Gain at K
        How good is our ranking overall?
    """
    def dcg_at_k(scores, k):
        scores = scores[:k]
        if len(scores) == 0:
            return 0.0
        return sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(scores))

    retrieved_k = retrieved_titles[:k]
    retrieved_scores = []

    # Use fuzzy matching to find relevance scores
    for retrieved_title in retrieved_k:
        score = 0.0
        for test_title, rel_score in relevance_scores.items():
            if is_fuzzy_match(test_title, retrieved_title):
                score = rel_score
                break
        retrieved_scores.append(score)

    dcg = dcg_at_k(retrieved_scores, k)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved_titles, relevant_titles, k):
    """Did we find at least one relevant result?"""
    retrieved_k = retrieved_titles[:k]
    for retrieved_title in retrieved_k:
        for test_title in relevant_titles:
            if is_fuzzy_match(test_title, retrieved_title):
                return 1.0
    return 0.0


# ============================================
# EVALUATION FUNCTIONS
# ============================================


def evaluate_query(index, query_data, k=10):
    """Evaluate a single query"""
    print("\n" + "="*80)
    print(f"QUERY: {query_data['query']}")
    print("="*80)

    # Search
    query_embedding = get_embedding(query_data['query'])
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
        filter={"rating": {"$gte": query_data.get('min_rating', 0.0)}}
    )

    retrieved_titles = [m['metadata']['title'] for m in results['matches']]
    relevant_titles = query_data['relevant_titles']
    relevance_scores = query_data['relevance_scores']

    # Show results
    print(f"\nTop {min(10, len(retrieved_titles))} Results:")
    print("-" * 80)
    for i, match in enumerate(results['matches'][:10], 1):
        title = match['metadata']['title']
        rating = match['metadata']['rating']

        is_relevant = False
        matched_test = None
        for test_title in relevant_titles:
            if is_fuzzy_match(test_title, title):
                is_relevant = True
                matched_test = test_title
                break

        is_relevant_icon = "✓" if is_relevant else "✗"
        rel_score = relevance_scores.get(matched_test, 0) if matched_test else 0

        if rel_score > 0:
            print(f"  {i+1}. {is_relevant_icon} {title} ({rating:.1f}, Rel:{rel_score}/5)")
            if matched_test and matched_test.lower() not in title.lower():
                print(f"      └─ Fuzzy matched: '{matched_test}'")
        else:
            print(f"  {i+1}. {is_relevant_icon} {title} ({rating:.1f})")

    # Calculate metrics
    print("\nMETRICS:")
    print("-" * 80)

    rr = reciprocal_rank(retrieved_titles, relevant_titles)
    ndcg = ndcg_at_k(retrieved_titles, relevance_scores, k)
    hit = hit_rate_at_k(retrieved_titles, relevant_titles, k)

    print(f"  Reciprocal Rank: {rr:.3f}")
    if rr > 0:
        print(f"    → First relevant at position {int(1/rr)}")

    print(f"\n  NDCG@{k}: {ndcg:.3f}")
    if ndcg >= 0.8:
        print("    Excellent!")
    elif ndcg >= 0.6:
        print("    Good")
    else:
        print("    Needs improvement")

    print(f"\n  Hit Rate@{k}: {hit:.1f}")

    return {
        "query": query_data['query'],
        "rr": rr,
        "ndcg": ndcg,
        "hit": hit
    }


# ============================================
# LLM AUTOMATED EVALUATION FUNCTIONS
# ============================================


def llm_relevance_judge(query: str, retrieved_title: str, retrieved_rating: float,
                        model: str = "gpt-4o-mini"):
    """
    Use an LLM to judge relevance of a retrieved result
    Returns: (is_relevant, relevance_score, reasoning)
    """
    prompt = f"""Given the search query and movie result, evaluate how relevant this movie is to the query.

    Search Query: "{query}"

    Retrieved Movie:
    - Title: {retrieved_title}
    - User Rating: {retrieved_rating}/5.0

    Rate the relevance on a scale of 0-5:
    5 = Highly relevant (perfect match for the query)
    4 = Very relevant (good match, meets query intent well)
    3 = Moderately relevant (acceptable match, somewhat related)
    2 = Slightly relevant (weak connection to query)
    1 = Barely relevant (minimal connection)
    0 = Not relevant (no meaningful connection)

    Provide your response in exactly this format:
    SCORE: [0-5]
    REASONING: [one sentence explanation]
    """

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic responses
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()

        # Parse response
        lines = content.split('\n')
        score_line = [line for line in lines if 'SCORE:' in line.upper()][0]
        reasoning_line = [line for line in lines if 'REASONING:' in line.upper()][0]

        score = float(score_line.split(':')[1].strip())
        reasoning = reasoning_line.split(':', 1)[1].strip()
        # Threshold: >3 is considered relevant
        is_relevant = score >= 3

        return is_relevant, score, reasoning

    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return False, 0.0, f"Error: {str(e)}"


def evaluate_with_llm_judge(index, queries, get_embedding_func,
                            k: int = 10, model: str = "gpt-4o-mini",
                            add_delay: bool = True):
    """
    Evaluate semantic search using LLM as judge

    Args:
        index: Pinecone index
        queries: List of queries to evaluate
        get_embedding_func: Function to generate embeddings
        k: Number of results to retrieve per query
        model: OpenAI model to use (gpt-4o-mini is cheaper, gpt-4o is better)
        add_delay: Add delay between API calls to avoid rate limits
    """
    results = []

    print(f"\n{'='*80}")
    print(f"LLM-as-Judge Evaluation (Model: {model})")
    print(f"{'='*80}\n")

    for query in tqdm.tqdm(queries, desc="Evaluating Queries"):

        # Get semantic search results
        query_embedding = get_embedding_func(query)
        res = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        relevant_items = []
        relevance_scores = {}
        judgments = []

        # Judge each retrieved result
        for rank, match in enumerate(res['matches'], 1):
            title = match['metadata']['title']
            rating = match['metadata']['rating']
            similarity_score = match['score']

            # Get LLM judgment
            is_rel, rel_score, reasoning = llm_relevance_judge(
                query, title, rating, model=model
            )

            if is_rel:
                relevant_items.append(title)
            relevance_scores[title] = rel_score

            # Store judgment
            judgments.append({
                'rank': rank,
                'title': title,
                'rating': rating,
                'similarity_score': similarity_score,
                'relevance_score': rel_score,
                'is_relevant': is_rel,
                'reasoning': reasoning
            })

            # Print result
            relevance_icon = "✓" if is_rel else "✗"
            print(f"  {rank}. {relevance_icon} [{rel_score}/5] {title}")
            print(f"      Rating: {rating:.1f} | Similarity: {similarity_score:.3f}")
            print(f"      {reasoning}")

            # Add delay to avoid rate limits
            if add_delay and rank < len(res['matches']):
                time.sleep(0.5)  # 500ms delay

        # Calculate metrics
        retrieved_titles = [m['metadata']['title'] for m in res['matches']]

        ndcg = ndcg_at_k(retrieved_titles, relevance_scores, k)
        mrr = reciprocal_rank(retrieved_titles, relevant_items)

        query_results = {
            "query": query,
            "retrieved_count": len(retrieved_titles),
            "relevant_count": len(relevant_items),
            f"ndcg@{k}": ndcg,
            "reciprocal_rank": mrr,
            "judgments": judgments
        }

        results.append(query_results)

        # Print metrics for this query
        print("\n  Query Metrics:")
        print(f"    Relevant Results: {len(relevant_items)}/{k}")
        print(f"    NDCG@{k}: {ndcg:.3f}")
        print(f"    Reciprocal Rank: {mrr:.3f}\n\n")

    # Calculate aggregate metrics
    print(f"\n{'='*80}")
    print("AGGREGATE METRICS (Across All Queries)")
    print(f"{'='*80}")

    avg_ndcg = sum(r[f"ndcg@{k}"] for r in results) / len(results)
    mrr_score = sum(r["reciprocal_rank"] for r in results) / len(results)

    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.3f}")
    print(f"Average NDCG@{k}: {avg_ndcg:.3f}")

    return results


# Helper function to save results
def save_results(results, filename: str):
    """Save evaluation results to a JSON file"""
    import json

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {filename}")
