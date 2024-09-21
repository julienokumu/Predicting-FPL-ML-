# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the csv into Pandas DataFrame
data = pd.read_csv('players.csv')

# Data Preprocessing
# Fill missing values
data.fillna(0, inplace=True)

# Convert categorical columns to numerical if necessary
# data = pd.get_dummies(data, columns = ['categorical_columm'])

# Identify and drop non-numeric columns
data_numeric = data.select_dtypes(include=['number'])

# If 'total_points' is not included in the numeric DataFrame
if 'total_points' not in data_numeric.columns:
    data_numeric['total_points'] = data['total_points']

# Separate features and target
features = data_numeric.drop(columns = ['total_points'])
target = data_numeric['total_points']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Choose a model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Make Predictions
all_predictions = model.predict(features)
data['predicted_points'] = all_predictions * 10

# Get the top 11 players based on the predicted points
top_players = data.nlargest(11, 'predicted_points')
print(top_players[['name', 'predicted_points']])

# Save csv file
top_players[['name', 'predicted_points']].to_csv('top11FPLPred.csv', index=False)