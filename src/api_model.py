# api_model.py
import os
import shutil
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import io
import pandas as pd
from utils import f1_score 
from typing import Optional, List
import tempfile
import mimetypes

# Initial setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn_v1.keras')
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'model_predictions.log')
PERFORMANCE_CSV = os.path.join(BASE_DIR, 'src', 'model_performance.csv')
IMAGE_SIZE = (96, 96)  # Width and height
INPUT_SHAPE = (*IMAGE_SIZE, 3)  # Adding channels
SUPPORTED_MIMETYPES = [
    'image/jpeg',
    'image/png',
    'image/tiff',
    'image/bmp',
    'image/webp'
]

# Create directories if they don't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PERFORMANCE_CSV), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Histopathologic Cancer Detection API",
    description="API for classifying histopathologic cancer images using CNN",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_image_file(file: UploadFile):
    """Validate the uploaded file is a supported image type"""
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp']:
        raise ValueError("Unsupported file extension")
    
    # Check MIME type
    file_mimetype = mimetypes.guess_type(file.filename)[0]
    if file_mimetype not in SUPPORTED_MIMETYPES:
        raise ValueError(f"Unsupported MIME type: {file_mimetype}")
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset pointer
    if file_size > max_size:
        raise ValueError(f"File too large. Max size is {max_size/1024/1024}MB")

def process_tiff_image(image_bytes: bytes) -> Image.Image:
    """Special handling for TIFF images"""
    try:
        # Save to temp file as some TIFF libraries need file path
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        # Open with PIL
        image = Image.open(tmp_path)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return image
    except Exception as e:
        raise ValueError(f"TIFF processing error: {str(e)}")

def preprocess_image(image_bytes: bytes, filename: str) -> np.ndarray:
    """Preprocess the image for model prediction with format-specific handling"""
    try:
        # Special handling for TIFF
        if filename.lower().endswith(('.tif', '.tiff')):
            image = process_tiff_image(image_bytes)
        else:
            image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and normalize
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        
        # Validate shape
        if image_array.shape != INPUT_SHAPE:
            raise ValueError(f"Invalid image shape. Expected {INPUT_SHAPE}, got {image_array.shape}")
        
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def log_prediction(image_id: str, prediction: str, confidence: float, processing_time: float):
    """Log prediction details with improved formatting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    log_entry = f"{timestamp},{image_id},{prediction},{confidence:.4f},{processing_time:.3f}\n"
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        logging.error(f"Failed to log prediction: {str(e)}")

# Load the model at startup with improved error handling
@app.on_event("startup")
async def load_model_on_startup():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        app.state.model = load_model(MODEL_PATH, custom_objects={'f1_score': f1_score})
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
        
    except Exception as e:
        logging.critical(f"Failed to load model: {str(e)}", exc_info=True)
        app.state.model = None  # Allow API to run without model for health checks

@app.post("/predict/", response_model=dict)
async def predict(file: UploadFile = File(...)):
    """Endpoint for cancer detection predictions with improved file handling"""
    start_time = datetime.now()
    
    try:
        # Validate file
        validate_image_file(file)
        contents = await file.read()
        
        # Preprocess and predict
        processed_image = preprocess_image(contents, file.filename)
        
        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
            
        prediction = app.state.model.predict(processed_image)
        confidence = float(prediction[0][0])
        result = "POSITIVE" if confidence > 0.5 else "NEGATIVE"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction
        image_id = os.path.splitext(file.filename)[0]
        log_prediction(image_id, result, confidence, processing_time)
        
        return {
            "status": "success",
            "result": result,
            "confidence": round(confidence, 4),
            "image_id": image_id,
            "processing_time_seconds": round(processing_time, 3),
            "model_version": "cnn_v1",
            "image_format": file.content_type,
            "image_size": f"{processed_image.shape[2]}x{processed_image.shape[1]}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error during prediction",
                "error": str(e)
            }
        )

@app.post("/batch_predict/", response_model=dict)
async def batch_predict(files: List[UploadFile] = File(...)):
    """Endpoint for batch predictions"""
    start_time = datetime.now()
    results = []
    
    try:
        if len(files) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
        
        for file in files:
            try:
                validate_image_file(file)
                contents = await file.read()
                processed_image = preprocess_image(contents, file.filename)
                
                prediction = app.state.model.predict(processed_image)
                confidence = float(prediction[0][0])
                result = "POSITIVE" if confidence > 0.5 else "NEGATIVE"
                
                results.append({
                    "image_id": os.path.splitext(file.filename)[0],
                    "result": result,
                    "confidence": round(confidence, 4),
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "image_id": os.path.splitext(file.filename)[0],
                    "result": None,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "completed",
            "results": results,
            "total_images": len(files),
            "successful_predictions": sum(1 for r in results if r['status'] == 'success'),
            "failed_predictions": sum(1 for r in results if r['status'] == 'failed'),
            "processing_time_seconds": round(processing_time, 3)
        }
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/", response_model=dict)
async def get_stats():
    """Endpoint for getting prediction statistics"""
    try:
        if not os.path.exists(LOG_FILE):
            return {
                "status": "success",
                "message": "No predictions recorded yet",
                "total_predictions": 0
            }
        
        df = pd.read_csv(LOG_FILE, 
                        names=['timestamp', 'image_id', 'prediction', 'confidence', 'processing_time'])
        
        total = len(df)
        positive = sum(df['prediction'] == 'POSITIVE')
        negative = total - positive
        
        return {
            "status": "success",
            "total_predictions": total,
            "positive_predictions": positive,
            "negative_predictions": negative,
            "positive_ratio": round(positive / total, 4) if total > 0 else 0,
            "average_confidence": round(df['confidence'].mean(), 4),
            "average_processing_time_seconds": round(df['processing_time'].mean(), 4),
            "last_prediction": df['timestamp'].iloc[-1] if total > 0 else None
        }
    except Exception as e:
        logging.error(f"Stats error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Comprehensive health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": hasattr(app.state, 'model') and app.state.model is not None,
        "disk_space": {
            "free_gb": round(shutil.disk_usage("/").free / (1024**3), 2),
            "total_gb": round(shutil.disk_usage("/").total / (1024**3), 2)
        },
        "system": {
            "cpu_cores": os.cpu_count(),
            "python_version": sys.version
        }
    }
    
    if not status["model_loaded"]:
        status["status"] = "degraded"
        status["message"] = "Model not loaded - predictions will fail"
    
    return status

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(BASE_DIR, 'logs', 'api.log')),
            logging.StreamHandler()
        ]
    )
    
    # Start server with improved configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,
        access_log=False,
        timeout_keep_alive=60
    )