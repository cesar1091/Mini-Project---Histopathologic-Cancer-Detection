# Histopathologic Cancer Detection

This project focuses on detecting cancerous tissue in histopathologic images using deep learning techniques. It includes a machine learning model and an API for serving predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
---

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Mini-Project---Histopathologic-Cancer-Detection
make install
```

## Usage

### Running the API

To start the API server, use the following command:

```bash
make run_api
```

This will:

- Start the API server using `uvicorn`.
- Host the server at `http://0.0.0.0:8000`.

You can access the API documentation (if using FastAPI) at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Project Structure

```
Mini-Project---Histopathologic-Cancer-Detection/
├── notebook/ Experiment.ipynb  # Experiment CNN classifier
├── src/ api_model.py          # API implementation (FastAPI or Flask)
├── models/                # Directory containing the trained model and related files
├── data/                 # Directory for datasets
├── [requirements.txt](http://_vscodecontentref_/1)      # List of dependencies
├── Makefile              # Automation commands
└── [README.md](http://_vscodecontentref_/2)             # Project documentation
```

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /health/`  
**Description:** Check the API status and system health  
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-04-02T12:00:00.000000",
  "model_loaded": true,
  "disk_space": {
    "free_gb": 45.67,
    "total_gb": 500.0
  },
  "system": {
    "cpu_cores": 8,
    "python_version": "3.11.0"
  }
}
```

### 2. Single Image Prediction
**Endpoint:** `POST /predict/`  
**Description:** Submit a single image for cancer detection  
**Parameters:**
- `file`: Image file (JPG, PNG, TIFF, BMP, WEBP) - max 10MB

**Response (Success):**
```json
{
  "status": "success",
  "result": "POSITIVE",
  "confidence": 0.8765,
  "image_id": "sample_image",
  "processing_time_seconds": 0.345,
  "model_version": "cnn_v1",
  "image_format": "image/tiff",
  "image_size": "96x96"
}
```

### 3. Batch Image Prediction
**Endpoint:** `POST /batch_predict/`  
**Description:** Submit multiple images (up to 20) for batch processing  
**Parameters:**
- `files`: List of image files (JPG, PNG, TIFF, BMP, WEBP) - max 10MB each

**Response (Success):**
```json
{
  "status": "completed",
  "results": [
    {
      "image_id": "image1",
      "result": "POSITIVE",
      "confidence": 0.9214,
      "status": "success"
    },
    {
      "image_id": "image2",
      "result": "NEGATIVE",
      "confidence": 0.1234,
      "status": "success"
    }
  ],
  "total_images": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "processing_time_seconds": 1.234
}
```

### 4. Statistics
**Endpoint:** `GET /stats/`  
**Description:** Get prediction statistics  
**Response:**
```json
{
  "status": "success",
  "total_predictions": 100,
  "positive_predictions": 65,
  "negative_predictions": 35,
  "positive_ratio": 0.65,
  "average_confidence": 0.7234,
  "average_processing_time_seconds": 0.456,
  "last_prediction": "2025-04-02 11:58:32.123456"
}
```

## Request/Response Examples

### cURL Examples

**Single Prediction:**
```bash
curl -X POST -F "file=@sample.tif" http://localhost:8000/predict/
```

**Batch Prediction:**
```bash
curl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  http://localhost:8000/batch_predict/
```

**Health Check:**
```bash
curl http://localhost:8000/health/
```

## Error Handling

The API returns appropriate HTTP status codes with JSON error messages:

| Status Code | Description | Example Response |
|-------------|-------------|------------------|
| 400 | Bad Request (invalid file type/size) | `{"detail": "Unsupported file extension"}` |
| 500 | Internal Server Error | `{"detail": "Internal server error during prediction"}` |
| 503 | Service Unavailable (model not loaded) | `{"detail": "Model not loaded"}` |

## Setup Instructions

1. **Prerequisites:**
   - Python 3.8+
   - pip

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the API:**
```bash
uvicorn api_model:app --reload
```

4. **Access the API:**
   - Docs: http://localhost:8000/docs
   - Redoc: http://localhost:8000/redoc

## Usage Examples

### Python Client Example

```python
import requests

# Single prediction
with open('sample.tif', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/',
        files={'file': f}
    )
print(response.json())

# Batch prediction
files = [('files', open('image1.jpg', 'rb')),
         ('files', open('image2.png', 'rb'))]
response = requests.post(
    'http://localhost:8000/batch_predict/',
    files=files
)
print(response.json())
```

## Supported Image Formats
- JPEG/JPG
- PNG
- TIFF/TIF
- BMP
- WEBP

**Note:** All images are automatically converted to RGB format and resized to 96x96 pixels.

## Dependencies

The project uses the following Python libraries:

- `tensorflow`
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `seaborn`
- `opencv-python`
- `Pillow`
- `scipy`
- `tqdm`
- `flask`
- `fastapi`
- `uvicorn`

For the full list, see `requirements.txt`.

