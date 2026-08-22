import pandas as pd
import numpy as np

# 1. Load the Dataset
df = pd.read_csv(r'C:\Users\PC\Desktop\cars.csv')

# 2. Inspect the Dataset
print(df.head())
print(df.dtypes)

# 3. Handle Missing Values
df['hp'].fillna(df['hp'].mean(), inplace=True)
df.dropna(subset=['hp'], inplace=True)

# 4. Filter Data
df_filtered = df[df['cyl'] == 6]
print(df_filtered)

# 5. Transform Columns
def categorize_hp(hp):
    if hp > 150:
        return 'High'
    elif hp > 80:
        return 'Medium'
    else:
        return 'Low'

df_filtered['HP_Category'] = df_filtered['hp'].apply(categorize_hp)

# 6. Save the Cleaned Dataset
df_filtered.to_csv(r'C:\Users\PC\Desktop\cleaned_cars_dataset.csv', index=False)

print("Data cleaning and preparation complete. The cleaned dataset is saved as 'cleaned_cars_dataset.csv'.")