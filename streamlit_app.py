import streamlit as st
import openai
from pinecone import Pinecone
# from dotenv import load_dotenv
from datetime import datetime


ENC_MODEL = "text-embedding-3-small"  # "text-embedding-ada-002"
EMB_DIM = 512  # dimensions: 1536
CONV_MAX_HIST = 3  # Length of conversation history to keep track of

# ============================================
# CONFIGURATION
# ============================================

# Set page config
st.set_page_config(
    page_title="Movie Recommendation Bot",
    page_icon="🎬",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/ConnieC14',
        'Report a Bug': 'mailto:cgc66@cornell.edu',
        'About': 'This is an agent assistant movie recommendation application.\nIt uses the movielens public dataset provided by Pinecone! :)'
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .conversation-stats {
        background-color: #e7f5e8;
        color: #1e5631;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4caf50;
    }
    .conversation-stats strong {
        color: #2e7d32;
    }
    .api-key-info {
        background-color: #fff3cd;
        color: #1e5631;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# API KEY CONFIGURATION
# ============================================


def check_api_keys():
    """Check if API keys are configured"""
    openai_key = st.session_state.get("openai_api_key", "")
    pinecone_key = st.session_state.get("pinecone_api_key", "")
    return bool(openai_key) and bool(pinecone_key)


# Show API key input if not configured
if not check_api_keys():
    st.title("🎬 Movie Recommendation Bot")
    st.markdown("### 🔑 API Configuration Required")

    st.markdown("""
    <div class="api-key-info">
        <strong>API Keys Required</strong><br>
        This app requires your own API keys to function. Your keys are stored only in your browser session
        and are never saved to any server.
    </div>
    """, unsafe_allow_html=True)

    with st.form("api_key_form"):
        st.markdown("#### Enter Your API Keys")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )

        pinecone_key = st.text_input(
            "Pinecone API Key",
            type="password",
            help="Get your API key from https://app.pinecone.io/"
        )

        index_name = st.text_input(
            "Pinecone Index Name",
            value="semantic-search-movie-demo",
            help="The name of your Pinecone index"
        )

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Start Chatting", use_container_width=True)
        with col2:
            if st.form_submit_button("How to Get API Keys", use_container_width=True):
                st.session_state.show_instructions = True

        if submit:
            if openai_key and pinecone_key:
                st.session_state.openai_api_key = openai_key
                st.session_state.pinecone_api_key = pinecone_key
                st.session_state.index_name = index_name
                st.rerun()
            else:
                st.error("Please enter both API keys to continue.")

    # Show instructions if requested
    if st.session_state.get("show_instructions", False):
        with st.expander("📖 How to Get Your API Keys", expanded=True):
            st.markdown("""
            ### OpenAI API Key
            1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign up or log in
            3. Click "Create new secret key"
            4. Copy the key (it starts with `sk-`)
            5. **Note:** You'll need to add credits to your OpenAI account

            ### Pinecone API Key
            1. Go to [Pinecone Console](https://app.pinecone.io/)
            2. Sign up or log in (free tier available)
            3. Go to "API Keys" in the left sidebar
            4. Copy your API key
            5. Create an index named `semantic-search-movie-demo` (or use your own name)

            ### Privacy Note
            Your API keys are stored only in your browser session and are never sent to any server
            except OpenAI and Pinecone for processing your requests.
            """)

    st.stop()  # Stop execution until keys are provided

# ============================================
# INITIALIZE CLIENTS (After API keys are set)
# ============================================


# Initialize APIs (use Streamlit secrets in production)
@st.cache_resource
def init_clients(openai_key, pinecone_key, index_name):
    try:
        openai.api_key = openai_key
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        return index, None
    except Exception as e:
        return None, str(e)


# Initialize with user-provided keys
index, error = init_clients(
    st.session_state.openai_api_key,
    st.session_state.pinecone_api_key,
    st.session_state.get("index_name", "semantic-search-movie-demo")
)

if error:
    st.error(f"Error connecting to APIs: {error}")
    st.info("💡 Please check your API keys in the sidebar and try again.")
    if st.button("Reset API Keys"):
        del st.session_state.openai_api_key
        del st.session_state.pinecone_api_key
        st.rerun()
    st.stop()

# Set OpenAI key for the current session
openai.api_key = st.session_state.openai_api_key

if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = {"embeddings": 0, "completions": 0}

# ============================================
# INITIALIZE SESSION STATE
# ============================================

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hai! I can assist you in finding relevant movies you might be interested in. Ask me about any genre, actor, or movie you're interested in!",
        "timestamp": datetime.now().isoformat(),
        "results": None
    })

# Track conversation metadata
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = datetime.now()

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ============================================
# HELPER FUNCTIONS
# ============================================


@st.cache_data(ttl=3600)  # 3600 seconds = 1 hour
def get_embedding(text):
    # embed text to vector
    response = openai.embeddings.create(
        model=ENC_MODEL,
        input=[text],
        dimensions=EMB_DIM
    )
    return response.data[0].embedding


def format_conversation_history(messages, max_history=5):
    """
    Format recent conversation history to give to LLM agent for context

    Args:
        messages: List of message dictionaries
        max_history: Maximum number of message pairs to include

    Returns:
        Formatted string of conversation history
    """
    # Get recent messages
    recent_messages = messages[-(max_history * 2):]

    if len(recent_messages) <= 1:
        return "This is the start of our conversation."

    history = []
    # Save all messages, but exclude the last message (current query)
    for msg in recent_messages[:-1]:
        if msg["role"] == "user":
            history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            history.append(f"Assistant: {msg['content']}")

    return "\n".join(history) if history else "This is the start of our conversation."


def generate_response(user_query, search_results,  conversation_history):
    """
    Generate response using conversation history for context

    Args:
        user_query: The user's question
        search_results: List of matches from our index database
        conversation_history: Formatted string of previous conversations

    Returns:
        Agent chatbot response
    """
    # Format our results
    results_context = ""
    for i, match in enumerate(search_results[:5], 1):
        title = match['metadata']['title']
        rating = match['metadata']['rating']
        results_context += f"{i}. {title} (Rating: {rating:.1f}/5.0)\n"

    # Build context-aware prompt
    prompt = f"""You are a helpful movie recommendation assistant having an ongoing conversation.

                CONVERSATION HISTORY:
                {conversation_history}

                USER QUESTION: "{user_query}"

                RELEVANT MOVIES FOUND:
                {results_context}

                INSTRUCTIONS:
                - Provide a friendly, conversational response
                - Reference a previous conversation if relevant (e.g., "Based on what you mentioned earlier about sci-fi...")
                - Mention the top 3-5 most relevant movies
                - Be Natural and engaging
                - Keep responses to 2-4 sentences.

                RESPONSE:"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )

    return response.choices[0].message.content.strip()


def save_conversation_to_file():
    """Export conversation history to a text file"""
    if len(st.session_state.messages) <= 1:
        return None

    conversation_text = "Movie Bot Conversation\n"
    conversation_text += f"Started: {st.session_state.conversation_started.strftime('%Y-%m-%d %H:%M:%S')}\n"
    conversation_text += f"Total Queries: {st.session_state.total_queries}\n"
    conversation_text += "="*60 + "\n\n"

    for msg in st.session_state.messages:
        timestamp = datetime.fromisoformat(msg["timestamp"]).strftime('%H:%M:%S')
        role = "You" if msg["role"] == "user" else "Bot"
        conversation_text += f"[{timestamp}] {role}: {msg['content']}\n"

        if msg.get("results"):
            conversation_text += "\n  Search Results:\n"
            for i, result in enumerate(msg["results"][:5], 1):
                conversation_text += f"    {i}. {result['title']} ⭐ {result['rating']:.1f}\n"
        conversation_text += "\n"

    return conversation_text

# ============================================
# UI
# ============================================


# Header
st.title("🎬 Movie Recommendation Bot")
st.markdown("*Ask me anything about movies and I'll find the best recommendations for you! :)*")

# Sidebar with settings
with st.sidebar:
    # API Key Status
    st.markdown("### 🔑 API Status")
    st.success("✓ Connected")
    with st.expander("⚙️ API Settings", expanded=False):
        st.text_input(
            "OpenAI Key",
            value=st.session_state.openai_api_key[:20] + "..." if len(st.session_state.openai_api_key) > 20 else st.session_state.openai_api_key,
            disabled=True,
            help="Your OpenAI API key (hidden for security)"
        )
        st.text_input(
            "Pinecone Key",
            value=st.session_state.pinecone_api_key[:20] + "..." if len(st.session_state.pinecone_api_key) > 20 else st.session_state.pinecone_api_key,
            disabled=True,
            help="Your Pinecone API key (hidden for security)"
        )
        if st.button("🔄 Change API Keys", use_container_width=True):
            del st.session_state.openai_api_key
            del st.session_state.pinecone_api_key
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()

    st.markdown("---")
    st.markdown("### 💰 Usage Tracker")
    st.write(f"Embeddings: {st.session_state.api_call_count['embeddings']}")
    st.write(f"Completions: {st.session_state.api_call_count['completions']}")
    est_cost = (st.session_state.api_call_count['embeddings'] * 0.0001 +
                st.session_state.api_call_count['completions'] * 0.0003)
    st.write(f"**Estimated cost:** ${est_cost:.4f}")

    st.header("⚙️ Settings")

    # Search Settings
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.5,
                           help="Only show movies with ratings above this threshold")
    top_k = st.slider("Number of Results", 5, 20, 10,
                      help="How many results to retrieve from the database")
    st.markdown("---")

    # Conversation Stats
    st.header("Conversation Stats 🗣️")
    st.markdown(f"""
    <div class="conversation-stats">
        <strong>Queries:</strong> {st.session_state.total_queries}<br>
        <strong>Messages:</strong> {len(st.session_state.messages)}<br>
        <strong>Started:</strong> {st.session_state.conversation_started.strftime("%B %d %H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

    # Export conversation
    if len(st.session_state.messages) > 1:
        conversation_export = save_conversation_to_file()
        if conversation_export:
            st.download_button(
                label="📥 Download Conversation",
                data=conversation_export,
                file_name=f"movielens-user-ratings_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.conversation_started = datetime.now()
        # Re-add greeting
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hai! I can assist you in finding relevant movies you might be interested in. Ask me about any genre, actor, or movie you're interested in!",
            "timestamp": datetime.now().isoformat(),
            "results": None
        })
        st.rerun()

    st.markdown("---")

    # Example queries
    st.header("💡 Example Queries")
    example_queries = [
        "What are the best sci-fi movies?",
        "Show me romantic comedies",
        "Find movies like Inception",
        "Best action films from the 2000s",
        "What horror movies would you recommend?"
    ]

    # Store questions in session state without immediate rerun
    for i, example in enumerate(example_queries):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            # Set the query to be processed in the next section
            st.session_state.pending_query = example

# ============================================
# UI - CHAT DISPLAY
# ============================================

# Display all messages in conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show search results if available
        if message.get("results"):
            with st.expander("📊 View Search Results", expanded=False):
                for i, result in enumerate(message["results"], 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{i}. {result['title']}**")
                    with col2:
                        st.write(f"⭐ {result['rating']:.1f}")
                    with col3:
                        st.write(f"🎯 {result['score']:.2f}")

# ============================================
# UI - CHAT INPUT
# ============================================
# Render user input box
prompt = st.chat_input("Ask about movies...")

# Check for any pending query from example button
if "pending_query" in st.session_state:
    prompt = st.session_state.pending_query
    del st.session_state.pending_query

# Chat input
if prompt:
    # Increment query counter
    st.session_state.total_queries += 1

    # keep track of api calls
    st.session_state.api_call_count["embeddings"] += 1
    st.session_state.api_call_count["completions"] += 1

    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat(),
        "results": None
    }
    st.session_state.messages.append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching for movies..."):
            try:
                # 1. Get embedding
                query_embedding = get_embedding(prompt)

                # 2. Query Pinecone
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"rating": {"$gte": min_rating}}
                )

                if results['matches']:
                    # 3. Format conversation history
                    conversation_history = format_conversation_history(
                        st.session_state.messages,
                        max_history=CONV_MAX_HIST
                    )

                    # 4. Generate context-aware response
                    response = generate_response(
                        prompt,
                        results['matches'],
                        conversation_history
                    )

                    st.markdown(response)

                    # 5. Prepare results data
                    results_data = [{
                        'title': m['metadata']['title'],
                        'rating': m['metadata']['rating'],
                        'score': m['score']
                    } for m in results['matches']]

                    # Show results in expander
                    with st.expander("📊 View Search Results", expanded=False):
                        for i, result in enumerate(results_data[:top_k], 1):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"**{i}. {result['title']}**")
                            with col2:
                                st.write(f"⭐ {result['rating']:.1f}")
                            with col3:
                                st.write(f"🎯 {result['score']:.2f}")

                    # Add assistant response to history
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                        "results": results_data[:top_k]
                    }
                    st.session_state.messages.append(assistant_message)

                else:
                    response = f"I couldn't find at least {top_k} movies with a rating above {min_rating:.1f} matching your criteria. Try adjusting the minimum rating or your top k results in the sidebar or rephrasing your query!"
                    st.markdown(response)

                    # Add to history without results
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                        "results": None
                    }
                    st.session_state.messages.append(assistant_message)

            except Exception as e:
                error_msg = f"Oh no! Something went wrong 🤯: {str(e)}"
                st.error(error_msg)

                # Add error to history
                assistant_message = {
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "results": None
                }
                st.session_state.messages.append(assistant_message)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "<a href='https://www.linkedin.com/in/consuelocuevas/' target='_blank' style='color: #667eea; text-decoration: none;'>Consuelo Cuevas</a> · "
    "App built using OpenAI, Pinecone & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
