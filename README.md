# NYC Airbnb Price Forecaster

A Streamlit application that predicts Airbnb prices in New York City based on property features and location.

## Features

- Interactive price prediction based on:
  - Neighborhood
  - Room type
  - Minimum nights
- Interactive map visualization of properties
- 7-day price forecast
- Real-time price predictions

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py              # Streamlit application
├── src/
│   └── model.py        # Model training and prediction
├── data/
│   └── clean/         # Cleaned data files
├── models/            # Trained model files
└── requirements.txt   # Project dependencies
```

## Development

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn

## License

MIT License 