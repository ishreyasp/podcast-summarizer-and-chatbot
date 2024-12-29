import os
import streamlit as st
import pickle
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from psycopg2.extras import RealDictCursor
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()

db_params = {
    'dbname': 'podcast_metadata',
    'user': 'airflow',
    'password': 'airflow',
    'host': '172.18.0.3',
    'port': '5432'
}

@st.cache_resource
def get_db_connection():
    return psycopg2.connect(**db_params)

def fetch_podcast_data(category):
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
        SELECT title, published, summary, sentiment, category
        FROM podcast_metadata
        WHERE category = %s;
        """
        cursor.execute(query, (category,))
        results = cursor.fetchall()
    return results

@st.cache_resource
def initialize_llm():
    return GoogleGenerativeAI(
        model=os.getenv('GOOGLE_LLM_MODEL'),
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.7
    )

def display_podcast(podcast):
    with st.container():
        st.subheader(podcast['title'])
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write('Published:', podcast['published'])
            st.write('Summary:', podcast['summary'])
        with col2:
            st.write('Sentiment:', podcast['sentiment'])
            st.write('Category:', podcast['category'])
        st.markdown("---")

def process_user_query(user_input, llm):
    if os.path.exists(os.getenv('PICKLE_FILE_PATH')):
        with open(os.getenv('PICKLE_FILE_PATH'), "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            result = chain({"question": user_input})
            return result
    return None

def main():
    st.set_page_config(layout="wide", page_title="Podcast Summary Viewer")
    llm = initialize_llm()
    st.title('Podcast Summary Viewer and Chatbot')
    categories = ["General", "Tech", "Finance"]
    selected_category = st.selectbox(
        'Filter by Category:', 
        categories, 
        index=categories.index("General")
    )
    podcast_data = fetch_podcast_data(category=selected_category)
    col1, col2 = st.columns([2, 1])    
    with col1:
        if podcast_data:
            st.write(f"Showing {len(podcast_data)} podcasts in {selected_category} category")
            for podcast in podcast_data:
                display_podcast(podcast)
        else:
            st.info(f"No podcasts found in the {selected_category} category.")
    with col2:
        st.sidebar.title('Podcast Chatbot')
        st.sidebar.write("Ask questions about the podcasts!")
        
        user_input = st.sidebar.text_input("Your question:")
        if st.sidebar.button('Submit'):
            if user_input:
                st.sidebar.write("You asked:", user_input)
                result = process_user_query(user_input, llm)
                
                if result:
                    st.sidebar.write("Response:", result["answer"])
                    sources = result.get("sources", "")
                    if sources:
                        st.sidebar.subheader("Sources:")
                        sources_list = sources.split("\n")
                        for source in sources_list:
                            if source.strip():
                                st.sidebar.write(source)
                else:
                    st.sidebar.error("Unable to process your question. Please try again.")
            else:
                st.sidebar.warning("Please enter a question.")

if __name__ == '__main__':
    main()