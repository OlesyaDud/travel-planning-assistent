import os
import json
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI


# Load environment variables
load_dotenv()
client = OpenAI()

# Initialize Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")  # Or use SUPABASE_SERVICE_KEY if needed
supabase = create_client(url, key)

# Load JSON data
with open("mock_hotels2.json") as f:
    data = json.load(f)

for record in data:
    embedding = client.embeddings.create(
        input=record["content"],
        model="text-embedding-3-small"
    ).data[0].embedding
    record['embedding'] = embedding

    response = supabase.table("travel_data").insert(record).execute()

    # Check success robustly
    if (hasattr(response, "status_code") and response.status_code in (200, 201)) \
       or (response.data and not (isinstance(response.data, dict) and "error" in response.data)):
        print(f"Inserted record with content: {record.get('content', 'N/A')}")
    else:
        print(f"Failed to insert record: {response.data}")

print("Finished uploading data.")