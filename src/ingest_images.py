import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

# CHANGE THIS to your dataset location (local)
LOCAL_DATASET = Path("C:/Users/yghad/Downloads/Data and Reference Code/assignment/data/brain_tumor_dataset")

# ADLS container
CONTAINER_NAME = "raw"
PREFIX = "tumor_images"   # will become: raw/tumor_images/yes and raw/tumor_images/no

def upload_images():
    storage_account_url = os.environ["AZURE_STORAGE_ACCOUNT_URL"]
    storage_account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

    print("📡 Connecting to Azure Storage...")
    blob_service = BlobServiceClient(
        account_url=storage_account_url,
        credential=storage_account_key
    )

    # Get the container client
    container = blob_service.get_container_client(CONTAINER_NAME)

    print("📁 Checking container...")
    try:
        container.create_container()
        print("🆕 Container created.")
    except Exception:
        print("✔️ Container already exists.")

    print("📤 Uploading dataset...")

    for label in ["yes", "no"]:
        local_path = LOCAL_DATASET / label
        for img_file in local_path.iterdir():

            # ADLS blob path
            blob_path = f"{PREFIX}/{label}/{img_file.name}"
            blob_client = container.get_blob_client(blob_path)

            # Skip existing files
            if blob_client.exists():
                print(f"⏩ Skipped (already exists): {blob_path}")
                continue

            with open(img_file, "rb") as f:
                blob_client.upload_blob(f)

            print(f"✅ Uploaded: {blob_path}")

    print("\n🎉 Ingestion complete!")

if __name__ == "__main__":
    upload_images()
