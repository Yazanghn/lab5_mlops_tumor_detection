# src/register_data_asset.py

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os

def register_tumor_images_data_asset():
    """Register the raw tumor images as a data asset in Azure ML Workspace"""
    
    # Connect to the workspace using the config.json file
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    
    # Create and register the data asset
    data_asset = Data(
        name="tumor_images_raw",
        description="Raw MRI brain tumor images from the bronze layer containing both yes and no folders",
        path="azureml://datastores/workspaceblobstore/paths/tumor_images",
        type=AssetTypes.URI_FOLDER
    )
    
    # Register the data asset
    registered_asset = ml_client.data.create_or_update(data_asset)
    
    print(f"Data asset registered successfully:")
    print(f"Name: {registered_asset.name}")
    print(f"Version: {registered_asset.version}")
    print(f"Full reference: azureml:{registered_asset.name}:{registered_asset.version}")
    print(f"Path: {registered_asset.path}")
    
    return registered_asset

if __name__ == "__main__":
    try:
        register_tumor_images_data_asset()
    except Exception as e:
        print(f"Error occurred: {e}")