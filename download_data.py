import os
import gdown

def download_data():
    target_name = "cached_features.zip"
    
    if os.path.exists(target_name):
        print(f"✅ {target_name} existe déjà.")
        return

    file_id = "1N3pnx-Wx4-G99sXm8i-yIUa3AkAw18ZQ"
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"🚀 Downloading {target_name} in progress... (2.5 GB, grab a coffee ☕)")
    gdown.download(url, target_name, quiet=False)
    print("\n✨ Download complete!")

if __name__ == "__main__":
    download_data()