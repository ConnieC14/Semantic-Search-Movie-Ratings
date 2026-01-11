
# Movie Recommendation Chat Agent

 AI movie recommendation chatbot integrated with semantic search vector database. After setting this up you will be able to ask and receive personalized suggestions for movie recommendations based on query similarity!

## Try the Live App

**[Launch Movie Recommendation Bot →](https://semantic-search-movie-ratings-a5wfzrzgn2kee4atvhgbti.streamlit.app/)**

> **Note:** You'll need to set up your own Pinecone index first (see below) and provide your API keys to use the app.

## Prerequisites

Before you begin, you'll need:

1. **OpenAI API Key** - Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Pinecone API Key** - Get it from [Pinecone Console](https://app.pinecone.io/)
3. **Python 3.8+** installed on your system

## Quick Start

### Step 1: Install Dependencies

```bash
    python -m venv venv 
    source venv/bin/activate
    pip install -r requirements.txt
    deactivate # when finished
```

### Step 2: Create Environment File

Create a file named `.env` in the same directory as `setup_pinecone_index.py`:

```env
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
```

**Important:** Never commit this file to GitHub! It's already included in `.gitignore`.

### Step 3: Run the Setup Script

```bash
python setup_pinecone_index.py
```

The script will:
- Load the MovieLens dataset
- Create embeddings for each movie
- Upload embeddings to Pinecone
- Test the index with sample queries

## Configuration

You can customize the setup by editing these variables in `setup_pinecone_index.py`:

```python
# OpenAI configuration
ENC_MODEL = "text-embedding-3-small"  # Embedding model
EMB_DIM = 512  # Dimensions (512 for speed, 1536 for quality)

# Pinecone configuration
INDEX_NAME = 'semantic-search-movie-demo'  # Your index name
CLOUD_PROVIDER = 'aws'  # aws, gcp, or azure
REGION = 'us-east-1'  # Deployment region

# Processing configuration
BATCH_SIZE = 100  # Vectors per batch
MAX_MOVIES = 1000  # Limit dataset (None for all ~1M movies)
```

## What Gets Stored

For each movie, the script stores:

```python
{
    "id": "tt4007502", # IMDB_id
    "values": [0.123, -0.456, ...],  # 512-dim embedding vector
    "metadata": {
        "title": "Deadpool 2 (2018)",
        "rating": 3.5
    }
}
```

## Next Steps

After setup is complete:

1. Check your `INDEX_NAME` (e.g., `semantic-search-movie-demo`)
2. Use this index name in the Streamlit app

## Testing the Index

The script automatically tests your index with these queries:
- "What was the movie titled Whiplash's rating?"
- "Find the best action movies from the 1990s"
- "Show me highly rated sci-fi films"

You should see relevant results with similarity scores.

## Troubleshooting

### Error: "OPENAI_API_KEY not found"
- Make sure your `.env` file exists in the same directory
- Check that the API key is correctly formatted (starts with `sk-`)

### Error: "Rate limit exceeded"
- OpenAI has rate limits. The script includes delays, but you may need to:
  - Wait a few minutes and try again
  - Reduce `BATCH_SIZE` to slow down requests
  - Upgrade your OpenAI plan

### Error: "Index not ready"
- Pinecone indexes take time to initialize (usually <1 minute)
- The script waits automatically, but you can increase the sleep time

### Error: "Dimension mismatch"
- If you change `EMB_DIM`, you must delete and recreate the index
- In Pinecone console: Delete index → Run script again

## Project Structure

```
semantic-search-movie-ratings/
├── setup_pinecone_index.py  # Setup script
├── streamlit_app.py   # Your App
├── requirements.txt   # Dependencies
├── .env               # API keys
└── README.md
```

## Security Notes

1. **Never commit `.env` files** - They contain your API keys
2. **Use environment variables** in production
3. **Rotate API keys** if they're exposed
4. **Monitor API usage** to avoid unexpected charges

## Cost Estimation

### For 1,000 movies (recommended for testing):
- **OpenAI Embeddings:** ~$0.01 (1,000 texts × ~10 tokens × $0.0001/1K tokens)
- **Pinecone:** Free tier (up to 100K vectors)
- **Total:** ~$0.01

### For 100,000 movies:
- **OpenAI Embeddings:** ~$1.00
- **Pinecone:** Free tier (up to 100K vectors)
- **Total:** ~$1.00

### For all ~1M movies (full dataset):
- **OpenAI Embeddings:** ~$10.00
- **Pinecone:** Paid plan required (~$70/month for pod-based)
- **Total:** $10 + monthly Pinecone costs

## Additional Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Quickstart](https://docs.pinecone.io/docs/quickstart)
- [MovieLens Dataset Info](https://docs.pinecone.io/guides/data/use-public-pinecone-datasets)
- [Other Datasets](https://github.com/erikbern/ann-benchmarks?tab=readme)

## License

This project uses the MovieLens dataset, which is provided by GroupLens Research at the University of Minnesota. Please see [MovieLens Data Usage](https://grouplens.org/datasets/movielens/) for terms and conditions.

