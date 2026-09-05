# Basic info
print(df.info())
print(f"\nFake news count: {df['label'].value_counts()[0]}")
print(f"Real news count: {df['label'].value_counts()[1]}")

# Check for missing values
print(f"\nMissing values:
{df.isnull().sum()}")