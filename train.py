import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import joblib

print("Loading massive dataset...")
df = pd.read_csv("G:\\ML\\Crop Recomendation system\\Crop_recommendation.csv") 

X = df.drop(columns=['label'])
y = df['label']

print("Configuring 10-Fold Cross-Validation...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

print("Training model (this might take a while)...")
nb_grid = GridSearchCV(GaussianNB(), param_grid, cv=kf, scoring='accuracy', n_jobs=1)
nb_grid.fit(X, y)


print("\n--- MANUAL TEST ---")

# Example values (you can change these)
N = 13
P = 34
K = 25
temperature = 38.39
humidity = 36.6
ph = 2.53
rainfall = 0.082

test_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

prediction = nb_grid.predict(test_input)[0]

print("Predicted Crop:", prediction)


# Extract your optimized final model
best_nb_model = nb_grid.best_estimator_
print(f"Training complete! Best accuracy score achieved.")

# SAVE THE TRAINED MODEL TO A FILE
joblib.dump(best_nb_model, 'naive_bayes_crop.pkl')
print("Saved trained model asset safely to 'naive_bayes_crop.pkl'!")
