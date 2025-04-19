"""
Only run this script when you want to rebuild the Pinecone index (e.g., when CV changes).
"""
import os
import time
from dotenv import load_dotenv
from utils import read_doc, chunk_data_sectionwise, chunk_data
from embeddings import SentenceTransformerEmbedding
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

# Load multilingual embedding model
embedding_model = SentenceTransformerEmbedding(
    SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=cloud, region=region)

# =====================
# Index TRINIDAD
# =====================
cv_path = r'docs/Trinidad_Monreal_Resume.pdf'
linkedin_path = r'docs/Trinidad_Monreal_LinkedIn_Profile.pdf'

cv_doc = read_doc(cv_path)
linkedin_doc = read_doc(linkedin_path)
trinidad_chunks = chunk_data_sectionwise(docs=(cv_doc + linkedin_doc), chunk_size=1000, chunk_overlap=100)

for i,chunk in enumerate(trinidad_chunks):
    print("Chunked Trinidad CV and LinkedIn:")
    print(f"Chunk {i+1}: {chunk.page_content[:50]}...")  # Print first 50 characters of each chunk

pc_index = 'tmonreal'
namespace = 'espacio'

if pc_index in pc.list_indexes().names():
    pc.delete_index(pc_index)
    print(f"Index {pc_index} deleted")

pc.create_index(pc_index, 
                dimension=384, 
                metric='cosine', 
                spec=spec)
print(f"Index created: {pc_index}")

# Upsert vectors
PineconeVectorStore.from_documents(
    documents=trinidad_chunks,
    index_name=pc_index,
    embedding=embedding_model,
    namespace=namespace
)

# Confirm upsert
index = pc.Index(pc_index)
for _ in range(5):
    stats = index.describe_index_stats()
    count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
    if count >= len(trinidad_chunks):
        print(f"✅ {count} vectors inserted into index'{pc_index}")
        break
    print("⏳ Waiting for vector sync...")
    time.sleep(2)
else:
    print("❌ Timeout: vectors not confirmed.")

# =====================
# Index JORGE
# =====================
jorge_cv_path = r'docs/CV_JORGE_CEFERINO_VALDEZ.pdf'
jorge_doc = read_doc(jorge_cv_path)
jorge_chunks = chunk_data(docs=jorge_doc, chunk_size=1500, chunk_overlap=100)

for i,chunk in enumerate(jorge_chunks):
    print("Chunked Jorge CV:")
    print(f"Chunk {i+1}: {chunk.page_content[:50]}...")  # Print first 50 characters of each chunk

pc_index = 'jvaldez'
if pc_index in pc.list_indexes().names():
    pc.delete_index(pc_index)
    print(f"Index {pc_index} deleted")

pc.create_index(pc_index, 
                dimension=384, 
                metric='cosine', 
                spec=spec)
print(f"Jorge index created: {pc_index}")

# Upsert vectors
PineconeVectorStore.from_documents(
    documents=jorge_chunks,
    index_name=pc_index,
    embedding=embedding_model,
    namespace=namespace
)

# Confirm upsert
index = pc.Index(pc_index)
for _ in range(5):
    stats = index.describe_index_stats()
    count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
    if count >= len(jorge_chunks):
        print(f"✅ {count} vectors inserted into index'{pc_index}'")
        break
    print("⏳ Waiting for vector sync...")
    time.sleep(2)
else:
    print("❌ Timeout: vectors not confirmed.")