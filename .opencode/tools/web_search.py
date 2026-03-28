import argparse
import http.client
import json
import os

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Search Google via Serper")
parser.add_argument("--query", "-q", type=str, help="Search query")
args = parser.parse_args()

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({"q": args.query})
headers = {
    "X-API-KEY": os.environ["SERPER_API_KEY"],
    "Content-Type": "application/json",
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = json.loads(res.read().decode("utf-8"))

organic = data.get("organic", [])
print(json.dumps(organic, indent=2, ensure_ascii=False))
