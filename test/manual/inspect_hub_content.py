"""Inspect SageMakerPublicHub content for the Llama 3.2 1B model."""
import boto3
import json

client = boto3.client("sagemaker")

# 1) Describe the model by its known JumpStart ID
print("=== describe_hub_content (JumpStart ID) ===")
resp = client.describe_hub_content(
    HubName="SageMakerPublicHub",
    HubContentType="Model",
    HubContentName="meta-textgeneration-llama-3-2-1b",
)
# Print key fields
for key in ["HubContentName", "HubContentDisplayName", "HubContentSearchKeywords", "HubContentDescription"]:
    print(f"{key}: {resp.get(key)}")

print("\n=== HubContentDocument (parsed) ===")
doc = json.loads(resp.get("HubContentDocument", "{}"))
print(json.dumps(doc, indent=2, default=str)[:3000])

# 2) List hub contents with NameContains to see what comes back
print("\n\n=== list_hub_contents (NameContains='Llama-32-1B') ===")
list_resp = client.list_hub_contents(
    HubName="SageMakerPublicHub",
    HubContentType="Model",
    NameContains="Llama-32-1B",
    MaxResults=10,
)
for s in list_resp.get("HubContentSummaries", []):
    print(f"  Name: {s['HubContentName']}")
    print(f"  Keywords: {s.get('HubContentSearchKeywords', [])}")
    print()

# 3) Try broader search
print("=== list_hub_contents (NameContains='llama-3-2-1b') ===")
list_resp2 = client.list_hub_contents(
    HubName="SageMakerPublicHub",
    HubContentType="Model",
    NameContains="llama-3-2-1b",
    MaxResults=10,
)
for s in list_resp2.get("HubContentSummaries", []):
    print(f"  Name: {s['HubContentName']}")
    print(f"  Keywords: {s.get('HubContentSearchKeywords', [])}")
    print()
