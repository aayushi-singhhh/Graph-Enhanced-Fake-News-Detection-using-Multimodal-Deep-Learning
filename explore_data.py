import pandas as pd

# Load the dataset
df = pd.read_csv("fake.csv")

print("Original dataset info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Type distribution:\n{df['type'].value_counts()}")
print("\n" + "="*50 + "\n")

# TODO: Clean dataset for fake news detection
# 1. Keep only 'title', 'text', and 'type' columns
df_clean = df[['title', 'text', 'type']].copy()
print("Step 1: Kept only 'title', 'text', and 'type' columns")
print(f"Shape after column selection: {df_clean.shape}")

# 2. Drop rows with missing values
print(f"Missing values before cleaning:\n{df_clean.isnull().sum()}")
df_clean = df_clean.dropna()
print(f"Shape after dropping missing values: {df_clean.shape}")

# 3. Combine 'title' and 'text' into a single column called 'content'
df_clean['content'] = df_clean['title'].astype(str) + " " + df_clean['text'].astype(str)
df_clean = df_clean[['content', 'type']]  # Keep only content and type
print("Step 3: Combined 'title' and 'text' into 'content' column")

# 4. Convert labels in 'type' to numeric (fake=0, real=1)
# Based on the data, we need to define what constitutes "fake" vs "real"
# Let's consider: fake, bias, conspiracy, hate, junksci as "fake" (0)
# And: bs, satire, state as "real" (1) - though this is debatable
print("Original type distribution:")
print(df_clean['type'].value_counts())

# Create binary classification: fake (0) vs real (1)
fake_categories = ['fake', 'bias', 'conspiracy', 'hate', 'junksci']
real_categories = ['bs', 'satire', 'state']

df_clean['label'] = df_clean['type'].apply(
    lambda x: 0 if x in fake_categories else 1 if x in real_categories else -1
)

# Remove any rows that don't fit our binary classification
df_clean = df_clean[df_clean['label'] != -1]
df_clean = df_clean[['content', 'label']]  # Keep only content and numeric label

print("\nStep 4: Converted to binary classification")
print("Label mapping: 0=fake (fake, bias, conspiracy, hate, junksci), 1=real (bs, satire, state)")
print(f"Label distribution:\n{df_clean['label'].value_counts()}")

# 5. Print dataset shape and first 5 rows
print("\n" + "="*50)
print("FINAL CLEANED DATASET:")
print(f"Dataset shape: {df_clean.shape}")
print("\nFirst 5 rows:")
print(df_clean.head())
