# api_model.py
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import csv
import io
from typing import Optional

# Initial setup
MODEL_PATH = '../models/cnn_v1.keras'
LOG_FILE = '../logs/model_predictions.log'
PERFORMANCE_CSV = '../src/model_performance.csv'
IMAGE_SIZE = (96, 96)  # Width and height
INPUT_SHAPE = (*IMAGE_SIZE, 3)  # Adding channels
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PERFORMANCE_CSV), exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="API de Detección de Cáncer Histopatológico",
    description="API para clasificación de imágenes de cáncer histopatológico usando un modelo CNN",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
@app.on_event("startup")
async def load_model_on_startup():
    try:
        app.state.model = load_model(MODEL_PATH)
        logging.info(f"Model loaded successfully. Input shape: {app.state.model.input_shape}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

def validate_image_format(image: Image.Image):
    """Validate the image meets our requirements"""
    if image.format not in ['JPEG', 'PNG']:
        raise ValueError("Only JPEG and PNG formats are supported")
    if image.mode != 'RGB':
        raise ValueError("Only RGB images are supported")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess the image for model prediction"""
    try:
        # Convert bytes to PIL image
        image = Image.open(io.BytesIO(image_bytes))
        validate_image_format(image)
        
        # Resize and normalize
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        
        # Validate shape
        if image_array.shape != INPUT_SHAPE:
            raise ValueError(f"Invalid image shape. Expected {INPUT_SHAPE}, got {image_array.shape}")
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def log_prediction(image_id: str, prediction: str, confidence: float, processing_time: float):
    """Log prediction details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{image_id},{prediction},{confidence:.4f},{processing_time:.3f}\n"
    
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        logging.error(f"Failed to log prediction: {str(e)}")

def calculate_metrics() -> dict:
    """Calculate performance metrics from log"""
    if not os.path.exists(LOG_FILE):
        return {"message": "No predictions recorded yet"}
    
    try:
        df = pd.read_csv(LOG_FILE, 
                        names=['timestamp', 'image_id', 'prediction', 'confidence', 'processing_time'])
        
        total = len(df)
        positive = sum(df['prediction'] == 'POSITIVO')
        negative = total - positive
        
        return {
            "total_predictions": total,
            "positive_predictions": positive,
            "negative_predictions": negative,
            "positive_ratio": round(positive / total, 4) if total > 0 else 0,
            "avg_confidence": round(df['confidence'].mean(), 4),
            "avg_processing_time": round(df['processing_time'].mean(), 4)
        }
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating metrics")

@app.post("/predict/", response_model=dict)
async def predict(file: UploadFile = File(...)):
    """Endpoint for making cancer detection predictions"""
    start_time = datetime.now()
    
    try:
        # Read and validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File too large")
        
        # Preprocess and predict
        processed_image = preprocess_image(contents)
        prediction = app.state.model.predict(processed_image)
        confidence = float(prediction[0][0])
        result = "POSITIVO" if confidence > 0.5 else "NEGATIVO"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction
        image_id = os.path.splitext(file.filename)[0]
        log_prediction(image_id, result, confidence, processing_time)
        
        return {
            "result": result,
            "confidence": round(confidence, 4),
            "image_id": image_id,
            "processing_time": round(processing_time, 3),
            "model_version": "cnn_v1"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/stats/", response_model=dict)
async def get_stats():
    """Endpoint for getting prediction statistics"""
    try:
        return calculate_metrics()
    except Exception as e:
        logging.error(f"Stats error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating statistics")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, 'model'),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/api.log'),
            logging.StreamHandler()
        ]
    )
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,
        access_log=False
    )