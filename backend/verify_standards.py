import os
import sys

# Add backend to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

try:
    from core.azure_blob_file_manager import azure_blob_file_manager
    
    def check_standards():
        container_name = "standards-documents"
        print(f"Checking container: {container_name}...")
        
        try:
            # Check connection first
            if not azure_blob_file_manager.is_connected():
                print("Connecting to Azure Blob Storage...")
                if not azure_blob_file_manager._setup_connection():
                    print("Failed to connect to Azure Blob Storage.")
                    return

            files = azure_blob_file_manager.list_files(container_name=container_name)
            
            if not files:
                print(f"WARNING: Container '{container_name}' appears to be empty or does not exist.")
                print("No standard documents found.")
            else:
                print(f"SUCCESS: Found {len(files)} files in '{container_name}':")
                for f in files:
                    print(f" - {f['name']} ({f['size']} bytes)")
                    
        except Exception as e:
            print(f"ERROR checking container: {e}")

    if __name__ == "__main__":
        check_standards()

except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you are running this from the backend directory and dependencies are installed.")
except Exception as e:
    print(f"Unexpected Error: {e}")
