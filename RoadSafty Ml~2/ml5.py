from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Prepare data for ML
X = df[['crime_rate', 'emergency_response_time', 'police_presence_encoded', 'street_lighting_encoded']]
y = df['safety_level']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Model MSE:", mean_squared_error(y_test, predictions))