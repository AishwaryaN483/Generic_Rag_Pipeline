from io import BytesIO
import os
import boto3
from dotenv import load_dotenv
from Trail.FSC.src.Parsing.parsepdf import parse_pdf

load_dotenv()

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not all([AWS_KEY, AWS_SECRET, AWS_REGION, S3_BUCKET_NAME]):
    raise EnvironmentError("Missing AWS environment variables")

    
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=AWS_REGION
)

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def read_pdf(bucket_name, object_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    pdf_bytes = response["Body"].read()

    parsed = parse_pdf(BytesIO(pdf_bytes))

    if "text" not in parsed:
        raise ValueError("Parsed PDF does not contain 'text' key")

    return parsed["text"]

OBJECT_KEY = "documents/1768732112455-Aishwarya_Nellutla.pdf"

text = read_pdf(S3_BUCKET_NAME, OBJECT_KEY)
chunks = chunk_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print()