import pandas as pd

# 1. Load CSV file
df = pd.read_csv("traffic_data.csv")

# 2. Keep only required columns
df = df[['date_time', 'traffic_volume']]

# 3. Convert date_time column to datetime format
df['date_time'] = pd.to_datetime(df['date_time'])

# 4. Sort data by date_time (important for time series)
df = df.sort_values('date_time')

# 5. Extract useful time features
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['day_of_week'] = df['date_time'].dt.dayofweek

# 6. Set date_time as index (good for time series models)
df.set_index('date_time', inplace=True)

# 7. Display processed data
print(df.head())