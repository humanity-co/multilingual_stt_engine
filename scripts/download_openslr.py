import os
import requests
import zipfile
import sys

DATA_ROOT = "../datasets"
os.makedirs(DATA_ROOT, exist_ok=True)

urls = {
    "mr_in_female": "https://www.openslr.org/resources/64/mr_in_female.zip"
}

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    # NOTE: In reality these datasets are large (gigabytes). We will only download
    # if it doesn't exist.
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return
        
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    mr_zip = os.path.join(DATA_ROOT, "mr_in_female.zip")
    mr_dir = os.path.join(DATA_ROOT, "marathi")
    
    download_file(urls["mr_in_female"], mr_zip)
    if not os.path.exists(mr_dir):
        os.makedirs(mr_dir, exist_ok=True)
        extract_zip(mr_zip, mr_dir)
        
    print("Marathi data prep ok.")
    
if __name__ == "__main__":
    main()
