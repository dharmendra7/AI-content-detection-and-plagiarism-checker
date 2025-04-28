import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import base64
import time
import io

from detect_text import detecting_text

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return ' '.join(tokens)

# Function to calculate cosine similarity
def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100  # Convert to percentage
    except:
        return 0

# Function to identify similar sentences
def find_similar_sentences(text1, text2, threshold=0.8):
    # Split texts into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    similar_pairs = []
    
    # Compare each sentence from text1 with each sentence from text2
    for i, sent1 in enumerate(sentences1):
        for j, sent2 in enumerate(sentences2):
            if len(sent1) < 10 or len(sent2) < 10:  # Skip very short sentences
                continue
                
            similarity = calculate_similarity(sent1, sent2)
            
            if similarity > threshold * 100:
                similar_pairs.append({
                    'original_sentence': sent1,
                    'comparison_sentence': sent2,
                    'similarity': similarity
                })
    
    return similar_pairs

# Function to generate downloadable report
def generate_report(original_text, comparison_text, overall_similarity, similar_sentences):
    buffer = io.StringIO()
    
    buffer.write("# Plagiarism Detection Report\n\n")
    buffer.write(f"## Overall Similarity: {overall_similarity:.2f}%\n\n")
    
    buffer.write("## Original Text\n\n")
    buffer.write(f"{original_text}\n\n")
    
    buffer.write("## Comparison Text\n\n")
    buffer.write(f"{comparison_text}\n\n")
    
    buffer.write("## Similar Sentences\n\n")
    
    if similar_sentences:
        for i, pair in enumerate(similar_sentences, 1):
            buffer.write(f"### Pair {i} - Similarity: {pair['similarity']:.2f}%\n\n")
            buffer.write(f"**Original:** {pair['original_sentence']}\n\n")
            buffer.write(f"**Comparison:** {pair['comparison_sentence']}\n\n")
    else:
        buffer.write("No significantly similar sentences found.\n\n")
    
    return buffer.getvalue()

# Function to create a download link
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Main application
def main():
    # Sidebar
    st.sidebar.title("📝 Plagiarism Detector")
    st.sidebar.markdown("---")
    
    # Add application modes
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Text Comparison", "Plagiarism Detection", "AI Content Detection"]
    )
    
    # Settings
    st.sidebar.markdown("## Settings")
    similarity_threshold = st.sidebar.slider("Similarity Threshold (%)", 50, 95, 80)


    if app_mode == "AI Content Detection":
        st.title("AI Content Detection")
        st.markdown("Detect AI-generated content")

        text_to_check = st.text_area("Enter text to Detect whethet it is AI generated or not:", height=200)
        if st.checkbox("Use sample text instead"):
            text_to_check = """The impact of artificial intelligence on modern society cannot be overstated. From healthcare to transportation, AI systems are revolutionizing how we live and work. While these technologies offer tremendous benefits, they also raise important ethical questions about privacy, bias, and the future of human labor. As we continue to develop increasingly sophisticated AI, it will be crucial to establish robust frameworks that ensure these tools enhance human welfare rather than diminish it."""
        
        # Add columns for button and options
        col1, col2 = st.columns([2, 1])
        with col1:
            analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        with col2:
            show_details = st.checkbox("Show detailed analysis", value=True)

        if analyze_button:
            if len(text_to_check) < 50:
                st.error("⚠️ Please enter more text for accurate analysis (at least 50 characters)")
            else:
                # Create a nice loading animation
                with st.spinner("🧠 Analyzing text patterns..."):
                    # Add a small artificial delay for visual effect
                    import time
                    time.sleep(0.5)
                    
                    # Get prediction
                    probability, predicted_label = detecting_text(text_to_check)
                    is_ai = predicted_label == 1
                    
                    # Calculate confidence levels
                    ai_confidence = probability * 100
                    human_confidence = 100 - ai_confidence
                    
                    # Create visual result display
                    st.markdown("### 📊 Analysis Results")
                    
                    # Create color and icon based on result
                    result_color = "#FF5733" if is_ai else "#33A1FF"
                    result_icon = "🤖" if is_ai else "👨‍💻"
                    result_label = "AI-Generated Content" if is_ai else "Human-Written Content"
                    
                    # Display main result with styling
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {result_color}20; border-left: 5px solid {result_color};">
                        <h2 style="color: {result_color}; margin:0;">{result_icon} {result_label}</h2>
                        <p style="font-size: 1.2em; margin-top: 10px;">Confidence: {ai_confidence:.1f}% likely to be AI-generated</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add confidence gauge visualization
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        # Display gauge chart for AI probability
                        st.markdown("#### AI vs Human Confidence")
                        st.progress(probability)
                        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Human: {human_confidence:.1f}%</span><span>AI: {ai_confidence:.1f}%</span></div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Simple explanation
                        st.markdown("#### What this means")
                        if ai_confidence > 90:
                            st.markdown("✅ **High confidence** this is AI-generated content")
                        elif ai_confidence > 70:
                            st.markdown("🔍 **Moderate confidence** this is AI-generated content")
                        elif ai_confidence > 50:
                            st.markdown("⚠️ **Low confidence** - could be AI or human")
                        elif ai_confidence > 30:
                            st.markdown("🔍 **Moderate confidence** this is human-written content")
                        else:
                            st.markdown("✅ **High confidence** this is human-written content")
                    
                    # Show detailed analysis if selected
                    if show_details:
                        st.markdown("### 🔬 Detailed Analysis")
                        
                        # Display text stats
                        word_count = len(text_to_check.split())
                        sent_count = text_to_check.count('.') + text_to_check.count('!') + text_to_check.count('?')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Word Count", word_count)
                        col2.metric("Sentences", sent_count)
                        col3.metric("Avg Words/Sentence", round(word_count/max(1, sent_count), 1))
                        
                        # Show explanation of how detection works
                        # st.markdown("""
                        # #### Detection Factors
                        
                        # AI detection analyzes patterns including:
                        # - Word choice and variety
                        # - Sentence structure and complexity
                        # - Stylistic consistency
                        # - Predictability of text patterns
                        
                        # **Note:** Detection is probabilistic and may not be 100% accurate.
                        # """)

    
    # Main area
    if app_mode == "Text Comparison":
        st.title("Text Comparison")
        st.markdown("Compare two texts to detect plagiarism")
        
        # Text input areas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            original_text = st.text_area("Enter the original text here:", height=200)
        
        with col2:
            st.subheader("Comparison Text")
            comparison_text = st.text_area("Enter the text to check for plagiarism:", height=200)
        
        # Check button
        if st.button("Check for Plagiarism"):
            if original_text and comparison_text:
                with st.spinner("Analyzing texts..."):
                    # Simulate a short delay to show the spinner
                    # time.sleep(1)
                    
                    # Calculate overall similarity
                    overall_similarity = calculate_similarity(original_text, comparison_text)
                    
                    # Find similar sentences
                    similar_sentences = find_similar_sentences(
                        original_text, 
                        comparison_text, 
                        threshold=similarity_threshold/100
                    )
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Create metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Overall Similarity", f"{overall_similarity:.2f}%")
                    col2.metric("Similar Sentences", len(similar_sentences))
                    
                    # Determine plagiarism level
                    if overall_similarity > 80:
                        col3.metric("Plagiarism Level", "High")
                        st.error("⚠️ High similarity detected! This may indicate significant plagiarism.")
                    elif 60 <= overall_similarity <= 80:
                        col3.metric("Plagiarism Level", "Moderate")
                        st.warning("⚠️ Moderate similarity detected. Some phrases may be copied.")
                    else:
                        col3.metric("Plagiarism Level", "Low")
                        st.success("✅ Low similarity. The texts appear to be distinct.")
                    
                    # Display similar sentences
                    if similar_sentences:
                        st.subheader(f"Similar Sentences ({len(similar_sentences)})")
                        
                        for i, pair in enumerate(similar_sentences, 1):
                            with st.expander(f"Pair {i} - Similarity: {pair['similarity']:.2f}%"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Original:**")
                                    st.info(pair['original_sentence'])
                                with col2:
                                    st.markdown("**Comparison:**")
                                    st.info(pair['comparison_sentence'])
                    
                    # Generate report
                    report = generate_report(original_text, comparison_text, overall_similarity, similar_sentences)
                    st.markdown(get_download_link(report, "plagiarism_report.md", "📥 Download Report"), unsafe_allow_html=True)
            else:
                st.error("Please enter both texts to compare")
    
    elif app_mode == "File Comparison":
        st.title("File Comparison")
        st.markdown("Upload two files to compare them for plagiarism")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original File")
            original_file = st.file_uploader("Upload original file", type=["txt", "pdf", "doc", "docx"])
        
        with col2:
            st.subheader("Comparison File")
            comparison_file = st.file_uploader("Upload file to check", type=["txt", "pdf", "doc", "docx"])
        
        if st.button("Compare Files"):
            if original_file and comparison_file:
                with st.spinner("Processing files..."):
                    # Currently only handling text files
                    try:
                        original_text = original_file.getvalue().decode("utf-8")
                        comparison_text = comparison_file.getvalue().decode("utf-8")
                        
                        # Calculate similarity
                        overall_similarity = calculate_similarity(original_text, comparison_text)
                        
                        # Find similar sentences
                        similar_sentences = find_similar_sentences(
                            original_text, 
                            comparison_text, 
                            threshold=similarity_threshold/100
                        )
                        
                        # Display results
                        st.subheader("Results")
                        
                        # Create metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Overall Similarity", f"{overall_similarity:.2f}%")
                        col2.metric("Similar Sentences", len(similar_sentences))
                        
                        # Determine plagiarism level
                        if overall_similarity > 80:
                            col3.metric("Plagiarism Level", "High")
                            st.error("⚠️ High similarity detected! This may indicate significant plagiarism.")
                        elif 60 <= overall_similarity <= 80:
                            col3.metric("Plagiarism Level", "Moderate")
                            st.warning("⚠️ Moderate similarity detected. Some phrases may be copied.")
                        else:
                            col3.metric("Plagiarism Level", "Low")
                            st.success("✅ Low similarity. The texts appear to be distinct.")
                        
                        # Generate report
                        report = generate_report(original_text, comparison_text, overall_similarity, similar_sentences)
                        st.markdown(get_download_link(report, "plagiarism_report.md", "📥 Download Report"), unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
                        st.info("Currently only text files are fully supported.")
            else:
                st.error("Please upload both files to compare")
    
    elif app_mode == "Plagiarism Detection":
        st.title("Plagiarism Detection")
        st.markdown("Compare text against web sources using Google Custom Search API")
        st.info("This feature searches the web for sources similar to your text and compares them.")
        
        text_to_check = st.text_area("Enter text to check against web sources:", height=200)
        
        # Add API key and Search Engine ID inputs (with option to hide)
        # show_api_settings = st.expander("API Settings", expanded=False)
        # with show_api_settings:
        api_key = "AIzaSyCzU8PptbZgYtqXOhFNDGsfQWIGm6C9OZc"
        search_engine_id = "25e90a6c42b9c445f"
        
        num_results = st.slider("Number of results to check:", min_value=1, max_value=10, value=5)
        
        if st.button("Check Against Web Sources"):
            if text_to_check:
                with st.spinner("Searching web sources..."):
                    # If no query is provided, use the first 50 characters of the text
                    
                    try:
                        # Import Google API client
                        from googleapiclient.discovery import build
                        import requests
                        from bs4 import BeautifulSoup as bs
                        
                        # Function to perform Google search
                        def google_search(search_term, api_key, search_engine_id=None, **kwargs):
                            """
                            Searches Google using the Custom Search JSON API.
                            """
                            try:
                                service = build("customsearch", "v1", developerKey=api_key)
                                
                                if search_engine_id:
                                    res = service.cse().list(q=search_term, cx=search_engine_id, **kwargs).execute()
                                else:
                                    res = service.cse().list(q=search_term, **kwargs).execute()
                                    
                                return res
                            except Exception as e:
                                st.error(f"API Error: {e}")
                                return None
                        
                        # Perform the search
                        results = google_search(text_to_check, api_key, search_engine_id, num=num_results)
                        
                        if not results or "items" not in results:
                            st.warning("No search results found. Try a different query or check your API credentials.")
                            urls = []
                        else:
                            # Extract URLs from search results
                            urls = [item['link'] for item in results.get("items", [])]
                            
                            # # Display search info
                            # st.subheader("Search Results")
                            # for i, item in enumerate(results.get("items", [])):
                            #     st.write(f"{i+1}. [{item['title']}]({item['link']})")
                            #     if 'snippet' in item:
                            #         st.write(f"   *{item['snippet']}*")
                            
                        # Set headers for web requests
                        headers = {
                            'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        # Function to extract text from a URL
                        def extract_text_from_url(url):
                            try:
                                response = requests.get(url, headers=headers, timeout=5)
                                soup = bs(response.text, 'html.parser')
                                
                                # Remove script and style elements
                                for script in soup(["script", "style"]):
                                    script.extract()
                                    
                                # Get text
                                text = soup.get_text()
                                
                                # Break into lines and remove leading and trailing space on each
                                lines = (line.strip() for line in text.splitlines())
                                # Break multi-headlines into a line each
                                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                                # Drop blank lines
                                text = '\n'.join(chunk for chunk in chunks if chunk)
                                
                                return text
                            except Exception as e:
                                return f"Error extracting text: {str(e)}"
                        
                        # Process each URL and calculate similarity
                        sources = []
                        similarities = []
                        
                        # Display progress bar
                        progress_bar = st.progress(0)
                        
                        for i, url in enumerate(urls):
                            progress_value = (i / len(urls)) if urls else 0
                            progress_bar.progress(progress_value)
                            
                            with st.status(f"Processing: {url}"):
                                source_text = extract_text_from_url(url)
                                if not source_text.startswith("Error"):
                                    # Calculate similarity using the function from your app
                                    similarity = calculate_similarity(text_to_check, source_text)
                                    
                                    # Store summary info and first 1000 chars of content
                                    sources.append({
                                        "url": url, 
                                        "text": source_text[:1000] + "..." if len(source_text) > 1000 else source_text,
                                        "similarity": similarity
                                    })
                                    similarities.append(similarity)
                                else:
                                    st.warning(f"Could not process {url}: {source_text}")
                        
                        # Complete the progress bar
                        progress_bar.progress(1.0)
                        
                        # Display results
                        if sources:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sources Checked", len(sources))
                            with col2:
                                highest_similarity = max(similarities) if similarities else 0
                                st.metric("Highest Similarity", f"{highest_similarity:.2f}%")
                            with col3:
                                avg_similarity = sum(similarities)/len(similarities) if similarities else 0
                                st.metric("Average Similarity", f"{avg_similarity:.2f}%")
                            
                            # Sort sources by similarity (highest first)
                            sources.sort(key=lambda x: x["similarity"], reverse=True)
                            
                            # Display each source with expandable details
                            st.subheader("Sources Found")
                            for idx, source in enumerate(sources):
                                with st.expander(f"Source {idx+1}: {source['url']} - Similarity: {source['similarity']:.2f}%"):
                                    st.write("Preview of content:")
                                    st.write(source["text"])
                                    
                                    # Show potentially matching paragraphs
                                    if source["similarity"] > 10:  # Only show for somewhat similar content
                                        st.subheader("Potential Matches")
                                        paragraphs = [p for p in source["text"].split("\n\n") if p.strip()]
                                        for p in paragraphs[:3]:  # Show first 3 paragraphs
                                            p_similarity = calculate_similarity(text_to_check, p)
                                            if p_similarity > 20:  # Show only if reasonably similar
                                                st.info(f"**{p_similarity:.2f}% similar**: {p}")
                                    
                                    st.write(f"[Visit source]({source['url']})")
                        else:
                            st.warning("No valid sources found. Try a different query.")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.markdown("""
                        ### Troubleshooting:
                        - Check your Google API key and Search Engine ID
                        - Verify your internet connection
                        - Try a different search query
                        - You may have exceeded your API quota
                        """)
            else:
                st.error("Please enter text to check against web sources")

    elif app_mode == "About":
        st.title("About Plagiarism Detector")
        
        st.markdown("""
        ## How it works
        
        This plagiarism detector uses several techniques to identify similarities between texts:
        
        1. **Text Preprocessing**:
           - Tokenization (breaking text into words)
           - Removing stopwords (common words like "the", "and", etc.)
           - Normalizing text (lowercase, removing punctuation)
        
        2. **TF-IDF Vectorization**:
           - Converts text into numerical vectors
           - Gives higher weight to important words
           - Reduces the importance of common words
        
        3. **Cosine Similarity**:
           - Measures the angle between text vectors
           - Ranges from 0% (completely different) to 100% (identical)
           - Independent of text length
        
        4. **Sentence-Level Comparison**:
           - Identifies specific sentences that may be plagiarized
           - Highlights matching content
        
        ## Limitations
        
        - **No External Database**: Currently compares only provided texts
        - **Language Support**: Primarily designed for English text
        - **Paraphrasing Detection**: May miss heavily paraphrased content
        - **Context Understanding**: Doesn't understand meaning, only patterns
        
        ## Future Improvements
        
        - Connect to academic databases and search engines
        - Implement more advanced NLP techniques
        - Add support for multiple languages
        - Improve paraphrase detection
        """)
        
        st.info("This application was created as a demonstration of plagiarism detection techniques using Python and Streamlit.")

if __name__ == "__main__":
    main()
