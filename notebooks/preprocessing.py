import pandas as pd

# Load dataset
df = pd.read_csv("data/startup_data.csv")  # Make sure the file name is correct

# Show shape
print(f"📊 Dataset Shape: {df.shape}")

# Check for missing values
print("\n🔍 Missing Values Per Column:")
print(df.isnull().sum())

# Check for duplicate rows
print("\n📦 Duplicate Rows:", df.duplicated().sum())

# Show column names and types
print("\n🧾 Column Names & Types:")
print(df.dtypes)

# Preview a few rows
print("\n🔎 Data Preview:")
print(df.head())
# Drop unwanted columns
columns_to_drop = [
    'Unnamed: 0', 'Unnamed: 6', 'id', 'object_id', 'state_code.1',
    'name', 'city', 'zip_code', 'labels', 'closed_at'
]
df.drop(columns=columns_to_drop, inplace=True)
print(f"\n🧹 Dropped {len(columns_to_drop)} unnecessary columns.")
# Fill milestone year nulls with median
df['age_first_milestone_year'].fillna(df['age_first_milestone_year'].median(), inplace=True)
df['age_last_milestone_year'].fillna(df['age_last_milestone_year'].median(), inplace=True)

print("\n✅ Filled milestone nulls with median values.")


# Check again for missing values
print("\n🔁 Missing values after drop:")
print(df.isnull().sum())

# Binary encode the target column
df['status'] = df['status'].apply(lambda x: 1 if x == 'acquired' or x == 'ipo' else 0)

print("\n🎯 Encoded target column 'status' → 1 = Success, 0 = Fail")

print("\n✅ Final Dataset Shape:", df.shape)
print("\n🔍 Final Data Sample:")
print(df.head())
