import json
import os
import pendulum
import requests
import xmltodict
import pickle
import google.generativeai as genai
import faiss
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAI
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment, effects 
from email.utils import parsedate_to_datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=2)
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

load_dotenv()

@dag(
    dag_id='podcast_summary',
    description='Podcast summarizer DAG',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=pendulum.datetime(2024, 12, 19),
    catchup=False
)
def podcast_summary():

    create_table = PostgresOperator(
        task_id='create_podcast_metadata_table',
        postgres_conn_id=os.getenv('POSTGRES_CONN_ID'),
        sql="""
            CREATE TABLE IF NOT EXISTS podcast_metadata (
                link TEXT PRIMARY KEY,
                title TEXT,
                filename TEXT,
                published TEXT,
                description TEXT,
                category VARCHAR(10),
                transcript TEXT,
                summary TEXT,
                sentiment VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
    )

    @task
    def get_podcast_metadata():
        gen_podcast_response = requests.get(os.getenv('GENERAL_PODCAST_URL'), headers=headers)
        if gen_podcast_response.status_code != 200:
            raise Exception(f"Failed to fetch data: HTTP status code {gen_podcast_response.status_code}")
        gen_feed = xmltodict.parse(gen_podcast_response.text)
        gen_episodes = gen_feed["rss"]["channel"]["item"]
        def parse_pub_date(episode):
            try:
                return parsedate_to_datetime(episode['pubDate'])
            except Exception as e:
                print(f"Error parsing pubDate: {episode['pubDate']} - {e}")
                return pendulum.datetime(1900, 1, 1)  
        sorted_gen_episodes = sorted(gen_episodes, key=parse_pub_date, reverse=True)
        recent_gen_episodes = sorted_gen_episodes[:2]

        tech_podcast_response = requests.get(os.getenv('TECH_PODCAST_URL'), headers=headers)
        if tech_podcast_response.status_code != 200:
            raise Exception(f"Failed to fetch data: HTTP status code {tech_podcast_response.status_code}")
        tech_feed = xmltodict.parse(tech_podcast_response.text)
        tech_episodes = tech_feed["rss"]["channel"]["item"]
        def parse_pub_date(episode):
            try:
                return parsedate_to_datetime(episode['pubDate'])
            except Exception as e:
                print(f"Error parsing pubDate: {episode['pubDate']} - {e}")
                return pendulum.datetime(1900, 1, 1)  
        sorted_tech_episodes = sorted(tech_episodes, key=parse_pub_date, reverse=True)
        recent_tech_episodes = sorted_tech_episodes[:2]

        fin_podcast_response = requests.get(os.getenv('FINANCE_PODCAST_URL'), headers=headers)
        if fin_podcast_response.status_code != 200:
            raise Exception(f"Failed to fetch data: HTTP status code {fin_podcast_response.status_code}")
        fin_feed = xmltodict.parse(fin_podcast_response.text)
        fin_episodes = fin_feed["rss"]["channel"]["item"]
        def parse_pub_date(episode):
            try:
                return parsedate_to_datetime(episode['pubDate'])
            except Exception as e:
                print(f"Error parsing pubDate: {episode['pubDate']} - {e}")
                return pendulum.datetime(1900, 1, 1)  
        sorted_fin_episodes = sorted(fin_episodes, key=parse_pub_date, reverse=True)
        recent_fin_episodes = sorted_fin_episodes[:2]

        recent_episodes = recent_gen_episodes + recent_fin_episodes + recent_tech_episodes
        print(f"Found {len(recent_episodes)} recent episodes.")
        return recent_episodes
    
    @task()
    def load_episodes(episodes):
        hook = PostgresHook(postgres_conn_id=os.getenv('POSTGRES_CONN_ID'))
        stored_episodes = hook.get_pandas_df("SELECT * FROM podcast_metadata;")
        new_episodes = []
        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                if episode['link'].split('/')[-2]=="marketplace":
                    category = "General"
                elif episode['link'].split('/')[-2]=="marketplace-tech":
                    category = "Tech"
                elif episode['link'].split('/')[-2]=="financially-inclined":
                    category = "Finance"    
                new_episodes.append(
                    [episode["link"], episode["title"], episode["pubDate"], episode["description"], filename, category]
                )
        if new_episodes:
            hook.insert_rows(table='podcast_metadata', rows=new_episodes,
                             target_fields=["link", "title", "published", "description", "filename", "category"])
        return [episode['link'] for episode in episodes]  
    
    @task()
    def download_episodes(episodes):
        os.makedirs(os.getenv('EPISODE_FOLDER'), exist_ok=True)
        for episode in episodes:
            name_end = episode["link"].split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(os.getenv('EPISODE_FOLDER'), filename)
            try:
                if not os.path.exists(audio_path):
                    print(f"Downloading episodes to: {os.path.abspath(os.getenv('EPISODE_FOLDER'))}")
                    audio = requests.get(episode["enclosure"]["@url"])
                    with open(audio_path, "wb+") as f:
                        f.write(audio.content)
            except Exception as e:
                print(f"An error occurred while downloading {filename}: {e}")

    def process_audio_and_store_in_db(file_name):
        try:
            print(f"Starting transcription for file: {file_name}")
            file_path = os.path.join(os.getenv('EPISODE_FOLDER'), file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            audio = AudioSegment.from_mp3(file_path)
            audio = effects.normalize(audio)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            chunks = [audio[i:i + 5 * 60 * 1000] for i in range(0, len(audio), 5 * 60 * 1000)]
            print("Loading Vosk model...")
            model = Model(model_path="/opt/airflow/vosk-models/vosk-model-en-us-0.22-lgraph")
            recognizer = KaldiRecognizer(model, 16000)
            recognizer.SetWords(True)
            transcript = ""
            for chunk in chunks:
                step = 20000  
                for i in range(0, len(chunk), step):
                    segment = chunk[i:i + step]
                    recognizer.AcceptWaveform(segment.raw_data)
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    transcript += text + " "
            print(f"Transcript length: {len(transcript)} characters")
            hook = PostgresHook(postgres_conn_id=os.getenv('POSTGRES_CONN_ID'))
            sql = """
                UPDATE podcast_metadata
                SET transcript = %s
                WHERE filename = %s;
            """
            hook.run(sql, parameters=(transcript.strip(), file_name))
            print(f"Transcript stored in database for file: {file_name}")

            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            raise

    @task
    def transcribe_audio_files():
        hook = PostgresHook(postgres_conn_id=os.getenv('POSTGRES_CONN_ID'))
        untranscribed_episodes = hook.get_pandas_df("SELECT * from podcast_metadata WHERE transcript IS NULL;")
        for index, row in untranscribed_episodes.iterrows():
            if row["filename"].endswith(".mp3"):
                print(f"Transcribing and storing transcript for {row['filename']}...")
                process_audio_and_store_in_db(row["filename"])
        print("All files processed and transcripts stored in the database successfully.")    

    @task
    def generate_summary():
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(os.getenv('GOOGLE_LLM_MODEL'))
        hook = PostgresHook(postgres_conn_id=os.getenv('POSTGRES_CONN_ID'))
        episodes_df = hook.get_pandas_df("SELECT filename, transcript FROM podcast_metadata WHERE summary IS NULL;")
        for index, row in episodes_df.iterrows():
            if row['filename'].endswith('.mp3'):
                print(f"Summarizing transcript for {row['filename']}...")
                prompt = f"""
                You are an AI assistant helping the user generate a 200-word short ingihtful summary for the following podcast transcript:
                '{row['transcript']}'
                """
                response = model.generate_content(prompt)
                summary = response.text
                print(summary)
                hook.run("UPDATE podcast_metadata SET summary = %s WHERE filename = %s;", parameters=(summary, row['filename']))

    @task
    def generate_sentiment_analysis():
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(os.getenv('GOOGLE_LLM_MODEL'))
        hook = PostgresHook(postgres_conn_id=os.getenv('POSTGRES_CONN_ID'))
        episodes_df = hook.get_pandas_df("SELECT filename, transcript FROM podcast_metadata WHERE sentiment IS NULL;")
        for index, row in episodes_df.iterrows():
            if row['filename'].endswith('.mp3'):
                prompt = f"""
                You are an AI assistant helping the user generate sentiment analysis for the following podcast transcript:
                '{row['transcript']}'
                Classify the results into Positive, Negative, or Neutral categories and give one word result only 
                like Positive, Negative or Neutral.
                """
                response = model.generate_content(prompt)
                sentiment = response.text
                print(sentiment)
                hook.run("UPDATE podcast_metadata SET sentiment = %s WHERE filename = %s;", parameters=(sentiment, row['filename']))

    @task
    def generate_embeddings():
        llm = GoogleGenerativeAI(
            model=os.getenv('GOOGLE_LLM_MODEL'),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.7
        )
        
        hook = PostgresHook(postgres_conn_id=os.getenv('POSTGRES_CONN_ID'))
        episodes_df = hook.get_pandas_df("SELECT filename, transcript FROM podcast_metadata WHERE transcript IS NOT NULL;")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        chunks_with_metadata = []
        for _, row in episodes_df.iterrows():
            chunks = text_splitter.split_text(row['transcript'])
            for chunk in chunks:
                chunks_with_metadata.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": row['filename']}  
                    )
                )
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv('HUGGINGFACE_EMBEDDING_MODEL'))
        embedding_size = len(embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_size)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(chunks_with_metadata)
        with open(os.getenv('PICKLE_FILE_PATH_FOR_DOCKER'), "wb") as f:
            pickle.dump(vector_store, f)
        print("All embeddings are stored and serialized in FAISS vector store.")    
        prompt_template = """Given the following question, generate an answer based on the vector store embeedings only.
                In the answer try to provide as much text as possible from the ebeddings as a source without making much changes.
                If the answer is not found, kindly state "I don't know." Don't try to make up an answer.

                QUESTION: What was US economy average growth in Q3?"""
        if os.path.exists(os.getenv('PICKLE_FILE_PATH_FOR_DOCKER')):
            with open(os.getenv('PICKLE_FILE_PATH_FOR_DOCKER'), "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                )
                result = chain({"question": prompt_template})
                print(result)

    podcast_metadata = get_podcast_metadata()
    create_table.set_downstream(podcast_metadata)
    episode_links = load_episodes(podcast_metadata)
    audio_files = download_episodes(podcast_metadata) 
    transcribe_episodes = transcribe_audio_files()
    podcast_summary = generate_summary()
    podcast_sentiment = generate_sentiment_analysis()
    transcript_embedding = generate_embeddings()

    audio_files >> transcribe_episodes >> [podcast_summary, podcast_sentiment, transcript_embedding]

summary = podcast_summary()