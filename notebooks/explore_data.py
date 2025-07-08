import pandas as pd

# ✅ FIXED path (no ../)
df = pd.read_csv("data/startup_data.csv")

# Preview
print(df.head())
print(df.info())
print(df.describe())
