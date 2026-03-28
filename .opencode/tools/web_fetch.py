import argparse
import http.client
import json
import os

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Fetch and scrape a webpage via Serper")
parser.add_argument("--url", "-u", type=str, help="URL to scrape")
args = parser.parse_args()

conn = http.client.HTTPSConnection("scrape.serper.dev")
payload = json.dumps({"url": args.url})
headers = {
    "X-API-KEY": os.environ["SERPER_API_KEY"],
    "Content-Type": "application/json",
}
conn.request("POST", "/", payload, headers)
res = conn.getresponse()
data = res.read().decode("utf-8")

print(data)
