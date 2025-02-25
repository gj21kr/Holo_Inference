import os 
import boto3 
import subprocess

def sync_s3_to_local(s3_path, local_path):
    cmd = f'aws s3 sync "{s3_path}" "{local_path}"'
    subprocess.run(cmd, shell=True, check=True)
    print(f"Downloaded {s3_path} to {local_path}")

def sync_local_to_s3(local_path, s3_path):
    cmd = f'aws s3 sync "{local_path}" "{s3_path}"'
    subprocess.run(cmd, shell=True, check=True)
    print(f"Uploaded {local_path} to {s3_path}")

def download_s3_to_local(s3_path, local_path):
    cmd = f'aws s3 cp "{s3_path}" "{local_path}" --recursive'
    subprocess.run(cmd, shell=True, check=True)
    print(f"Downloaded {s3_path} to {local_path}")

def upload_local_to_s3(local_path, s3_path):
    cmd = f'aws s3 cp "{local_path}" "{s3_path}" --recursive'
    subprocess.run(cmd, shell=True, check=True)
    print(f"Uploaded {local_path} to {s3_path}")

def download_weights_from_url(url, local_path):
    from niftynet.utilities.download import download_and_decompress
    download_and_decompress(url, local_path)

if __name__ == "__main__":
    # LargeIASeg / CMHA
    # brain_windowed / total_segmentation
    s3_path = 's3://jungeun/train_results/DIAS/Att_UNet_241101_140419/'
    local_path = '/home/xcath-ai/Data/train_results/DIAS/Att_UNet_241101_140419/'

    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)

    # download_s3_to_local(s3_path, local_path)
    # upload_local_to_s3(local_path, s3_path)
    # sync_local_to_s3(local_path, s3_path)
    sync_s3_to_local(s3_path, local_path)