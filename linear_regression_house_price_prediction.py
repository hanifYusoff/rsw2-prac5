import joblib
from pathlib import Path

# 1. Get the directory of the current script
script_dir = Path(__file__).parent

# 2. Build the absolute path to your model file
model_path = script_dir / "linear_regression_model.joblib"

# 3. Load the model safely
model = joblib.load(model_path)
