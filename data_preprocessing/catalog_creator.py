import pandas as pd

# Load both sources
img_df = pd.read_csv(r"D:\flickd\data\images.csv")
meta_df = pd.read_excel(r"D:\flickd\data\product_data.xlsx")

# Get first image per product_id
img_df = img_df.sort_values("id").drop_duplicates("id")
img_df = img_df.rename(columns={"image_url": "shopify_cdn_url"})

# Merge with metadata
catalog_df = pd.merge(meta_df, img_df, on="id")
catalog_df = catalog_df.rename(columns={"id": "product_id"})

# Save as master catalog
catalog_df.to_csv(r"D:\flickd\data_preprocessing\catalog.csv", index=False)