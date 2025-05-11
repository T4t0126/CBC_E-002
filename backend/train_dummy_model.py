import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Dummy training data
data = {
    'focus_time': [10, 20, 30, 40, 50],
    'distraction_events': [5, 4, 2, 1, 0],
    'emotion_score': [0.2, 0.4, 0.6, 0.8, 0.9],
    'behavior': ['Mischievous', 'Silent', 'Silent', 'Active', 'Active']
}

df = pd.DataFrame(data)

# Train a basic classifier
X = df[['focus_time', 'distraction_events', 'emotion_score']]
y = df['behavior']

model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('backend/engagement_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Dummy model saved as engagement_model.pkl")