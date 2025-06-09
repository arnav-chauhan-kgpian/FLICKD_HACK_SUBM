import pandas as pd
import nltk
import re
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # Add this import
from keybert import KeyBERT

# Load the catalog data
df = pd.read_csv(r"D:\flickd\data_preprocessing\catalog_with_paths.csv")
df = df.drop(columns=['alias', 'mrp', 'price_display_amount', 'discount_percentage'])

# Download required NLTK data
nltk.download('punkt', quiet=True)  # Changed from 'punkt_tab' to 'punkt'
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize NLTK components (ADD THESE LINES)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

print(f"Loaded {len(df)} products from catalog")

# 1. PRODUCT TAGS PREPROCESSING
# List of keys you want to keep (as per hackathon requirements)
keys_to_keep = ['Colour', 'Print']

def parse_product_tags(tag_string):
    """Parse product tags string into dictionary with only relevant keys"""
    if pd.isna(tag_string):
        return {}
    
    # Split by comma and strip whitespace
    parts = [part.strip() for part in tag_string.split(',')]
    # Build dict for only the keys you want
    return {
        k.strip(): v.strip() for part in parts if ':' in part
        for k, v in [part.split(':', 1)]
        if k.strip() in keys_to_keep
    }

df['product_tags'] = df['product_tags'].apply(parse_product_tags)

# Remove Gender key as it's not needed for matching (per hackathon guidelines)
def remove_key_from_dict(d, key):
    """Remove a specific key from dictionary"""
    if isinstance(d, dict):
        d = d.copy()
        d.pop(key, None)
        return d
    return d

df['product_tags'] = df['product_tags'].apply(lambda d: remove_key_from_dict(d, 'Gender'))

# 2. TITLE PREPROCESSING
def preprocess_title(title):
    """Clean and preprocess product titles"""
    if not isinstance(title, str):
        return ""

    # Split on '|' or assume brand is first word
    title = title.replace("|", " ").strip()
    words = title.split()

    # Remove the first word (brand name) as per hackathon guidelines
    if len(words) > 1:
        words = words[1:]

    # Join back and clean
    cleaned = " ".join(words).lower()
    cleaned = re.sub(r"[^a-z0-9\s\-]", "", cleaned)  # keep letters, digits, hyphens
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

df['title'] = df['title'].apply(preprocess_title)

# 3. PRODUCT COLLECTIONS PREPROCESSING
# Define blacklist of generic collection names (as identified from frequency analysis)
blacklist = {
    'flash sale', 'earth-friendly fabrics', 'kalki koechlin | shop the look', 'all collections',
    'casual essentials', 'natural fabrics', 'just in!', 'hidden gems', 'cotton collection', 'inventory',
    'dresses', 'tops', 'topwear', 'the hourglass closet', 'the figure', 'flat 20% edit', 'the pear closet',
    'hidden gems 1', 'the figure 8 closet', 'lounge wear', 'bottomwear veronica', 'all dresses',
    'topwear veronica', 'influencers edit-2', 'discount year end sale', 'the rectangle closet',
    'dresses veronica', 'content creator edit', 'creator edit', 'the apple closet', 'skirts | shorts',
    "mira's casual wardrobe", 'influencer edit', 'best sellers', 'unstoppably you!', 'viscose collection',
    'creators edit', 'shirts and blouses', 'discount delights', 'the virgio promise', 'essential closet',
    "creator's edit", 'the grid collective', 'tru linen', 'co-ords', 'rock & rosã©', 'the rakhi edit',
    "kinnari's top favourite", "kinnari's picks", 'upto 30% off', "mira's collection", 'fashion factor x virgio', 
    'perfect match!', 'linen luxe', 'top hourglass', 'dresses with pockets', 'miras style closet', 
    'premium picks', 'summer capsule', 'top 100 styles', "mounis picks", 'trousers', 'coord-sets', 
    "mira kapoor's picks", "soha's pop collection", "ruhani's wardrobe", 'recycled fabrics', "mouni's picks", 
    'day wear', 'new launch', 'collection', 'fast selling', 'eco blue dresses for women', 'perfect match',
    'azure isles', 'top selling products', 'best selling', "kinnari's wardrobe", "mira kapoor's mini picks",
    'cotton luxe', 'upto 40% off', 'cozy winter', 'new in', 'co-ord sets veronica', 
    'eco-friendly skirt co-ord sets for women', "soha's wardrobe", 'influencer picks', 'shorts', 
    'elevated essentials', "fatimas picks", 'knee length dresses', "pratibha rantas' picks", 'maxis', 
    'the holiday edit', 'jackets', 'plunge bra', 'the veronicas', "kalki's holiday vibe", 
    "the kaarigar's bazaar", 'concert collection', "mira's virgio wardrobe", "neha dhupia's picks!", 
    'min 30% off', 'christmas collection', 'new arrival', 'styles for content', 'summer soirã©e', 
    'day dress', "masoom's picks", 'evening wear', 'bottom wear', 'selling fast', 'pure linen', 
    'alaya f x elle', 'born to shine', 'china workwear', 'recycled polyester', 'shanaya', 'academic edge', 
    'diya', 't-shirt bra', 'tuesday fashion finds', 'raven', 'linen col.', 'everyday styles',
    'reborn: the upcyled edit', 'all black series', 'stick on bra', 'bandeau bra', 'jumpsuits', 
    'vacay edit', 'spotted_new collection', 'avika', "soha's picks", 'circular sunday (04-02-2024)', 
    'shirt dresses for women - organic cotton', 'lyocell collection', 'xyz', 'josie', 
    'spotted_july collection', 'aishwarya kaushal', 'summer capsule', 'aishwaryaâ€™s selections'
}

def clean_collections_string(collection_str):
    """Clean product collections by removing generic/marketing terms"""
    if pd.isna(collection_str):
        return ''

    # Split, clean, and filter
    tags = [tag.strip().lower() for tag in collection_str.split(',') if tag.strip()]
    filtered_tags = [tag for tag in tags if tag not in blacklist]
    
    return ', '.join(filtered_tags)

df['product_collections'] = df['product_collections'].apply(clean_collections_string)

# 4. DESCRIPTION PREPROCESSING 
def preprocess_description(description):
    """
    Basic preprocessing for product description:
    - Lowercase
    - Remove punctuation and digits
    - Tokenize
    - Remove stopwords
    - Lemmatize tokens
    - Return cleaned string
    """
    if not isinstance(description, str) or not description.strip():
        return ""

    try:
        # Lowercase
        text = description.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        cleaned_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and len(token) > 2
        ]

        return ' '.join(cleaned_tokens)
    
    except Exception as e:
        print(f"Error processing description: {e}")
        return ""

df['description'] = df['description'].apply(preprocess_description)

# 5. FINAL VALIDATION AND CLEANUP
# Ensure all required columns are present for hackathon
required_columns = ['product_id', 'title', 'description', 'product_type', 'product_tags', 
                   'product_collections', 'shopify_cdn_url', 'local_path']

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: Missing required columns: {missing_columns}")

# Remove rows with missing critical data
initial_count = len(df)
df = df.dropna(subset=['product_id', 'title', 'product_type'])
final_count = len(df)

if initial_count != final_count:
    print(f"Removed {initial_count - final_count} rows with missing critical data")

# 6. SAVE PROCESSED CATALOG
output_path = r"D:\flickd\data_preprocessing\catalog_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"✅ Preprocessing complete!")
print(f"📊 Final catalog: {len(df)} products")
print(f"💾 Saved to: {output_path}")

# Display sample of processed data
print("\n📋 Sample of processed data:")
print(df[['product_id', 'title', 'description', 'product_collections']].head(3))

# Show statistics
print(f"\n📈 Statistics:")
print(f"- Products with collections: {df['product_collections'].str.len().gt(0).sum()}")
print(f"- Products with descriptions: {df['description'].str.len().gt(0).sum()}")
print(f"- Unique product types: {df['product_type'].nunique()}")
