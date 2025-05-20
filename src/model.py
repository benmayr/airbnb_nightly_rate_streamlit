import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json

def load_clean_data(path: str) -> pd.DataFrame:
    """Load and clean the data using the schema from step 1."""
    df = pd.read_csv(path)
    
    # Convert price to numeric, handling both string and numeric formats
    if pd.api.types.is_string_dtype(df['price']):  # if price is string
        df['price'] = pd.to_numeric(df['price'].str.replace('$', '').str.replace(',', ''))
    else:  # if price is already numeric
        df['price'] = pd.to_numeric(df['price'])
    
    return df

def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features and target for the model."""
    # Create features
    features = pd.DataFrame()
    
    # Define all possible categories
    all_neighbourhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    all_room_types = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
    
    # One-hot encode neighbourhood_group and room_type
    neighbourhood_dummies = pd.get_dummies(df['neighbourhood_group'], prefix='neighbourhood')
    room_dummies = pd.get_dummies(df['room_type'], prefix='room')
    
    # Ensure all categories are present
    for neighbourhood in all_neighbourhoods:
        col_name = f'neighbourhood_{neighbourhood}'
        if col_name not in neighbourhood_dummies.columns:
            neighbourhood_dummies[col_name] = 0
            
    for room_type in all_room_types:
        col_name = f'room_{room_type}'
        if col_name not in room_dummies.columns:
            room_dummies[col_name] = 0
    
    # Combine dummies
    features = pd.concat([neighbourhood_dummies, room_dummies], axis=1)
    
    # Bucket minimum_nights
    features['min_nights_1'] = (df['minimum_nights'] == 1).astype(int)
    features['min_nights_2_6'] = ((df['minimum_nights'] >= 2) & 
                                 (df['minimum_nights'] <= 6)).astype(int)
    features['min_nights_7plus'] = (df['minimum_nights'] >= 7).astype(int)
    
    # Add other features
    features['availability_365'] = df['availability_365']
    features['latitude'] = df['latitude']
    features['longitude'] = df['longitude']
    
    # Create target if price column exists
    if 'price' in df.columns:
        y = np.log1p(df['price'])
        return features, y
    else:
        return features, None

def train_random_forest(X, y, *, random_state=42) -> Pipeline:
    """Train a random forest model with preprocessing pipeline."""
    # Create preprocessing steps
    numeric_features = ['availability_365', 'latitude', 'longitude']
    categorical_features = [col for col in X.columns if col.startswith(('neighbourhood_', 'room_', 'min_nights_'))]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    # Train model
    pipeline.fit(X, y)
    return pipeline

def evaluate(model, X_test, y_test) -> dict:
    """Evaluate model performance using MAE and SMAPE."""
    y_pred = model.predict(X_test)
    
    # Calculate MAE
    mae = np.mean(np.abs(y_pred - y_test))
    
    # Calculate SMAPE
    smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test)))
    
    return {
        'mae': mae,
        'smape': smape
    }

def save(model, path="models/rf_price.pkl"):
    """Save the trained model using joblib."""
    joblib.dump(model, path)

if __name__ == "__main__":
    # Load and prepare data
    df = load_clean_data("data/clean/listings.csv")
    X, y = make_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    print("Model Performance:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"SMAPE: {metrics['smape']:.2f}%")
    
    # Save model
    save(model)