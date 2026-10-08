import os
import warnings
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
from trendspy import Trends
from sentence_transformers import SentenceTransformer, util
import redditwarp.SYNC
import redditwarp.models.submission

# --- ENVIRONMENT & WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="Newsroom Multi-Platform Content Gap Dashboard",
    page_icon="🗞️",
    layout="wide"
)

CBC_RSS_FEEDS = [
"https://www.cbc.ca/webfeed/rss/rss-topstories", "https://www.cbc.ca/webfeed/rss/rss-world", "https://www.cbc.ca/webfeed/rss/rss-canada", "https://www.cbc.ca/webfeed/rss/rss-politics", "https://www.cbc.ca/webfeed/rss/rss-business", "https://www.cbc.ca/webfeed/rss/rss-health", "https://www.cbc.ca/webfeed/rss/rss-arts", "https://www.cbc.ca/webfeed/rss/rss-technology", "https://www.cbc.ca/webfeed/rss/rss-Indigenous", "https://www.cbc.ca/webfeed/rss/rss-sports", "https://www.cbc.ca/webfeed/rss/rss-sports-mlb", "https://www.cbc.ca/webfeed/rss/rss-sports-nba", "https://www.cbc.ca/webfeed/rss/rss-sports-curling", "https://www.cbc.ca/webfeed/rss/rss-sports-cfl", "https://www.cbc.ca/webfeed/rss/rss-sports-nfl", "https://www.cbc.ca/webfeed/rss/rss-sports-nhl", "https://www.cbc.ca/webfeed/rss/rss-sports-soccer", "https://www.cbc.ca/webfeed/rss/rss-sports-figureskating", "https://www.cbc.ca/webfeed/rss/rss-sports-golf", "https://www.cbc.ca/webfeed/rss/rss-sports-olympics", "https://www.cbc.ca/webfeed/rss/rss-sports-skiing", "https://www.cbc.ca/webfeed/rss/rss-sports-tennis", "https://www.cbc.ca/webfeed/rss/rss-canada-britishcolumbia", "https://www.cbc.ca/webfeed/rss/rss-canada-kamloops", "https://www.cbc.ca/webfeed/rss/rss-canada-calgary", "https://www.cbc.ca/webfeed/rss/rss-canada-edmonton", "https://www.cbc.ca/webfeed/rss/rss-canada-saskatchewan", "https://www.cbc.ca/webfeed/rss/rss-canada-saskatoon", "https://www.cbc.ca/webfeed/rss/rss-canada-manitoba", "https://www.cbc.ca/webfeed/rss/rss-canada-thunderbay", "https://www.cbc.ca/webfeed/rss/rss-canada-sudbury", "https://www.cbc.ca/webfeed/rss/rss-canada-windsor", "https://www.cbc.ca/webfeed/rss/rss-canada-london", "https://www.cbc.ca/webfeed/rss/rss-canada-kitchenerwaterloo", "https://www.cbc.ca/webfeed/rss/rss-canada-toronto", "https://www.cbc.ca/webfeed/rss/rss-canada-hamiltonnews", "https://www.cbc.ca/webfeed/rss/rss-canada-montreal", "https://www.cbc.ca/webfeed/rss/rss-canada-newbrunswick", "https://www.cbc.ca/webfeed/rss/rss-canada-pei", "https://www.cbc.ca/webfeed/rss/rss-canada-novascotia", "https://www.cbc.ca/webfeed/rss/rss-canada-newfoundland", "https://www.cbc.ca/webfeed/rss/rss-canada-north", "https://www.cbc.ca/webfeed/rss/rss-canada-ottawa"
]
# --- HELPER FUNCTIONS ---
def format_to_k(volume):
    if isinstance(volume, (int, float)):
        if volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M+"
        elif volume >= 1_000:
            return f"{volume / 1_000:.0f}K+"
    return str(volume)


# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --- DATA INGESTION: GOOGLE TRENDS ---
@st.cache_data(ttl=900)
def get_google_trends_data():
    try:
        tr = Trends()
        articles = []
        count = 0
        trends = tr.trending_now(geo='CA')
        for item in trends[:10]:
            count += 1
            news = tr.trending_now_news_by_ids(item.news_tokens, max_news=1)
            for article in news:
                articles.append({
                    'source': 'Google Trends',
                    'rank': count,
                    'trend_name': item.normalized_keyword,
                    'trend_search_volume': f"{format_to_k(item.volume)} searches",
                    'title': article.title,
                    'context_source': article.source,
                    'url': article.url
                })
        return articles
    except Exception as e:
        st.error(f"Error fetching Google Trends: {e}")
        return []


# --- DATA INGESTION: REDDIT ---
def process_reddit_post(reddit_posts, subm, count):
    if isinstance(subm, redditwarp.models.submission.LinkPost):
        flair = ""
        try:
            if subm.b.get('link_flair_richtext') and subm.b['link_flair_richtext'][0].get('e') == 'text':
                flair = subm.b['link_flair_richtext'][0].get('t', '')
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        domain = ""
        try:
            domain = subm.b.get('domain', '') if isinstance(subm.b, dict) else getattr(subm.b, 'domain', '')
        except AttributeError:
            pass

        reddit_posts.append({
            'source': 'Reddit',
            'rank': count,
            'trend_name': subm.title[:60] + "..." if len(subm.title) > 60 else subm.title,
            'trend_search_volume': f"{format_to_k(subm.score)} upvotes ({format_to_k(subm.comment_count)} comments)",
            'title': subm.title,
            'context_source': domain or 'Reddit Link',
            'url': subm.link,
            'reddit_permalink': f"https://reddit.com{subm.permalink}",
            'flair': flair
        })


@st.cache_data(ttl=900)
def get_reddit_trends_data(filter_type="hot"):
    try:
        client = redditwarp.SYNC.Client()
        reddit_posts = []
        it = (client.p.subreddit.pull.hot('Canada', amount=15) 
              if 'hot' in filter_type else 
              client.p.subreddit.pull.rising('Canada', amount=15))
        
        count = 0
        for subm in list(it):
            if isinstance(subm, redditwarp.models.submission.LinkPost):
                count += 1
                process_reddit_post(reddit_posts, subm, count)
                if count >= 10:
                    break
        return reddit_posts
    except Exception as e:
        st.error(f"Error fetching Reddit trends: {e}")
        return []


# --- DATA INGESTION: X / TWITTER ---
@st.cache_data(ttl=900)
def get_x_trends():
    url = "https://getdaytrends.com/canada/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching X trends: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    trends = []
    table = soup.select_one('table')

    if table:
        rows = table.select('tbody tr') if table.select('tbody tr') else table.find_all('tr')
        for row in rows:
            if row.find('th') and not row.select_one('th.pos'):
                continue
            rank, topic, tweets = None, None, "N/A"
            rank_tag = row.select_one('th.pos, td.pos, td:nth-of-type(1)')
            if rank_tag:
                try:
                    rank = int(rank_tag.text.strip().replace('.', ''))
                except ValueError:
                    continue
            topic_tag = row.select_one('td.main a, td a')
            if topic_tag:
                topic = topic_tag.text.strip()
            tweets_tag = row.select_one('.desc, .text-muted, td:nth-of-type(3)')
            if tweets_tag:
                tweets = tweets_tag.text.strip().split('\n')[0]

            if rank is not None and topic:
                search_query = urllib.parse.quote(topic)
                trends.append({
                    'source': 'X',
                    'rank': rank,
                    'trend_name': topic,
                    'trend_search_volume': tweets if tweets != "N/A" else "Volume Hidden",
                    'title': topic,
                    'context_source': 'X Trending',
                    'url': f"https://x.com/search?q={search_query}"
                })
                if len(trends) >= 10:
                    break
    return trends


# --- DATA INGESTION: CBC RSS FEEDS ---
def fetch_single_feed(feed_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(feed_url, headers=headers, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        return [{'title': entry.title, 'link': getattr(entry, 'link', '')} for entry in feed.entries]
    except Exception:
        return []


@st.cache_data(ttl=900)
def fetch_all_cbc_articles():
    seen_links = set()
    all_articles = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(fetch_single_feed, url): url for url in CBC_RSS_FEEDS}
        for future in as_completed(future_to_url):
            for article in future.result():
                if article['link'] and article['link'] not in seen_links:
                    seen_links.add(article['link'])
                    all_articles.append(article)
    return all_articles


# --- NLP PRE-COMPUTATION ENGINE ---
@st.cache_data(ttl=900)
def compute_all_scores(_model, trend_data_list, cbc_articles):
    if not cbc_articles or not trend_data_list:
        return []

    cbc_headlines = [a['title'] for a in cbc_articles]
    headline_embeddings = _model.encode(cbc_headlines, convert_to_tensor=True)

    precalculated_data = []

    for trend_obj in trend_data_list:
        combined_trend_context = f"{trend_obj['trend_name']} {trend_obj['title']}"
        trend_embedding = _model.encode(combined_trend_context, convert_to_tensor=True)

        cosine_scores = util.cos_sim(trend_embedding, headline_embeddings)[0]
        scores_list = [score.item() for score in cosine_scores]

        precalculated_data.append({
            "trend_obj": trend_obj,
            "scores": scores_list
        })

    return precalculated_data


# --- MAIN APP INTERFACE ---
st.title("🚨 Newsroom Multi-Platform Content Gap Dashboard")
st.markdown("Cross-referencing Canadian trends across **Google**, **Reddit**, and **X** against current **CBC News** coverage.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Controls & Filters")
    
    similarity_threshold = st.slider(
        "AI Match Sensitivity",
        min_value=0.40, max_value=0.80, value=0.55, step=0.05,
        help="Higher values require strict story matching. Lower values capture looser contextual matches."
    )
    
    st.divider()
    st.subheader("Platforms to Track")
    include_google = st.checkbox("Google Trends", value=True)
    include_reddit = st.checkbox("Reddit (r/Canada)", value=True)
    reddit_filter = st.radio("Reddit Sorting", ["hot", "rising"], horizontal=True)
    include_x = st.checkbox("X / Twitter", value=True)

    st.divider()
    if st.button("🔄 Force Refresh All Feeds"):
        st.cache_data.clear()

# Load Model
model = load_model()

# Fetch Active Ingestion Streams
with st.spinner("Fetching live trend data across platforms and parsing CBC feeds..."):
    google_data = get_google_trends_data() if include_google else []
    reddit_data = get_reddit_trends_data(filter_type=reddit_filter) if include_reddit else []
    x_data = get_x_trends() if include_x else []
    cbc_articles = fetch_all_cbc_articles()

all_platform_trends = google_data + reddit_data + x_data

# Display Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Active Platforms", sum([include_google, include_reddit, include_x]))
col2.metric("Total Trends Tracked", len(all_platform_trends))
col3.metric("CBC Articles Scanned", len(cbc_articles))
st.divider()

if not all_platform_trends:
    st.warning("No trends collected. Please check platform selections or refresh.")
    st.stop()

if not cbc_articles:
    st.error("Failed to load CBC RSS feeds.")
    st.stop()

# Run Cached NLP Pre-computation
with st.spinner("Calculating semantic similarities with Sentence-Transformers..."):
    precalculated_data = compute_all_scores(model, all_platform_trends, cbc_articles)

# Fast Threshold Filter Loop
gaps = []
covered = []

for item in precalculated_data:
    trend_obj = item["trend_obj"]
    scores = item["scores"]

    matching_cbc_articles = []
    for idx, score in enumerate(scores):
        if score >= similarity_threshold:
            matching_cbc_articles.append(cbc_articles[idx])

    if not matching_cbc_articles:
        gaps.append(trend_obj)
    else:
        covered.append({"trend": trend_obj, "matches": matching_cbc_articles})

# --- RENDER RESULTS ---
st.subheader(f"🔴 Content Gaps ({len(gaps)} Uncovered Topics)")
if not gaps:
    st.success("No gaps detected! CBC coverage matches all active trends.")

for gap in gaps:
    icon = "🔍" if gap['source'] == "Google Trends" else ("🔥" if gap['source'] == "Reddit" else "🐦")
    with st.container(border=True):
        st.error(f"**{icon} [{gap['source']}] #{gap['rank']} Trend:** {gap['trend_name']} ({gap['trend_search_volume']})")
        st.markdown(f"**Source Context:** [{gap['title']}]({gap['url']}) — *{gap['context_source']}*")
        if 'reddit_permalink' in gap:
            st.caption(f"[View Reddit Comments Thread]({gap['reddit_permalink']})")

st.markdown("---")

st.subheader(f"🟢 Covered Topics ({len(covered)} Matched)")
for cov in covered:
    tr = cov['trend']
    matches = cov['matches']
    icon = "🔍" if tr['source'] == "Google Trends" else ("🔥" if tr['source'] == "Reddit" else "🐦")
    with st.container(border=True):
        st.success(f"**{icon} [{tr['source']}] #{tr['rank']} Trend:** {tr['trend_name']} ({tr['trend_search_volume']})")
        st.markdown(f"**Source Context:** [{tr['title']}]({tr['url']}) — *{tr['context_source']}*")
        
        with st.expander(f"View {len(matches)} CBC Article Match(es)"):
            for m in matches:
                st.markdown(f"- [{m['title']}]({m['link']})")