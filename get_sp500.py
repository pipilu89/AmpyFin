import pandas as pd
# Load the CSV file into a DataFrame
df_sp500 = pd.read_csv('sp500.csv', encoding='ISO-8859-1')

# Display the first few rows of the DataFrame
print(df_sp500.head())

# Extract the 'Symbol' column and convert it to a list
sp500_symbols = df_sp500['Symbol'].tolist()

# Display the list of symbols
print(sp500_symbols)