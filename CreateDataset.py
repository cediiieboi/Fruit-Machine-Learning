import pandas as pd

# Create a sample dataset

data = {
    'Weight': [150, 120, 180, 170, 100, 200],
    'Color': ['Red', 'Yellow', 'Orange', 'Green', 'Yellow', 'Orange'],
    'Diameter': [7, 6, 8, 7, 5, 9],
    'Fruit Type': ['Apple', 'Banana', 'Orange', 'Apple', 'Banana', 'Orange']
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Convert to a CSV file
df.to_csv('sample_fruit_dataset.csv', index=False)

print("Working working working")