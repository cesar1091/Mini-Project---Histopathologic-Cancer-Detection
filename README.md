# Histopathologic Cancer Detection

This project focuses on detecting cancerous tissue in histopathologic images using deep learning techniques. It includes a machine learning model and an API for serving predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
- [Project Structure](#project-structure)
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

# Dependencies

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