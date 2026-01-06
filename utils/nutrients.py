import sys
import os
import pandas as pd

# Add the project root directory to sys.path to resolve the 'config' module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import settings
except ImportError:
    # Fallback in case the above fails
    sys.path.insert(0, project_root)
    from config import settings

def filter_csv_by_label(label_value):
    """
    Filters the nutrients CSV by the given label.
    """
    if not os.path.exists(settings.NUTRIENTS_PATH):
        raise FileNotFoundError(f"Nutrient data not found at: {settings.NUTRIENTS_PATH}")
        
    df = pd.read_csv(settings.NUTRIENTS_PATH)
    return df[df['label'].str.lower() == label_value.lower()]

if __name__ == "__main__":
    test_label = 'pizza'
    print(f"Filtering nutrients for: {test_label}")
    result = filter_csv_by_label(test_label)
    if not result.empty:
        print(result)
    else:
        print(f"No data found for label: {test_label}")
