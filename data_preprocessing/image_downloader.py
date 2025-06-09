import os
import pandas as pd
import requests
from tqdm import tqdm

df = pd.read_csv(r"D:\flickd\data_preprocessing\catalog.csv")
save_dir = r"D:\flickd\data_preprocessing\catalog_images.csv"
os.makedirs(save_dir, exist_ok=True)

local_paths = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = row["product_id"]
    url = row["shopify_cdn_url"]
    filename = f"{pid}.jpg"
    local_path = os.path.join(save_dir, filename)

    try:
        if not os.path.exists(local_path):
            r = requests.get(url, timeout=10)
            r.raise_for_status()  # Raise exception for bad status codes
            with open(local_path, "wb") as f:
                f.write(r.content)
        local_paths.append(local_path)
    except Exception as e:
        print(f"Failed to download image for product_id={pid}, url={url}. Error: {e}")
        local_paths.append("")

df["local_path"] = local_paths
df.to_csv(r"D:\flickd\data_preprocessing\catalog_with_paths.csv", index=False)

print(f"Download complete. Successful: {(df['local_path'] != '').sum()}, Failed: {(df['local_path'] == '').sum()}")
