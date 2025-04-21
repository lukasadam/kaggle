import os
import kagglehub
import zipfile
from pathlib import Path 
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Set the base path to the project directory
base_path = Path(__file__).parents[1]
# Set the path to the data directory
data_dir = base_path / "data"
data_dir.mkdir(exist_ok=True)

# Download competition data
competition = 'playground-series-s4e9'
competition_path = data_dir / competition
competition_path.mkdir(exist_ok=True)
api.competition_download_files(competition, path=competition_path)

# Download dataset data
dataset = 'taeefnajib/used-car-price-prediction-dataset'
dataset_path = data_dir / 'used_car_price_dataset'
dataset_path.mkdir(exist_ok=True)
api.dataset_download_files(dataset, path=dataset_path)

# Unzip the downloaded files
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

for zip_file in competition_path.glob('*.zip'):
    unzip_file(zip_file, competition_path)
for zip_file in dataset_path.glob('*.zip'):
    unzip_file(zip_file, dataset_path)

