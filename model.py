import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv("fertilizer.csv")

# Encode categorical data
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])
df['Crop_Type'] = le_crop.fit_transform(df['Crop_Type'])
df['Fertilizer'] = le_fertilizer.fit_transform(df['Fertilizer'])

X = df.drop('Fertilizer', axis=1)
y = df['Fertilizer']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump((model, le_soil, le_crop, le_fertilizer), open("model.pkl", "wb"))

print("Model trained and saved!")