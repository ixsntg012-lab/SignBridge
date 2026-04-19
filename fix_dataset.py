import pandas as pd

CSV_PATH = "data/signs.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

print("Original samples:", len(df))

# -----------------------------
# Remove rows with missing data
# -----------------------------
df = df.dropna()

# -----------------------------
# Keep only valid labels (a-z)
# -----------------------------
df = df[df['label'].str.match("^[a-z]$")]

# -----------------------------
# Remove duplicate rows
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# Reset index
# -----------------------------
df = df.reset_index(drop=True)

print("Cleaned samples:", len(df))

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(CSV_PATH, index=False)

print("Dataset cleaned and saved!")