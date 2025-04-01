install:
	pip install --upgrade pip && pip install -r requirements.txt

run_api:
    uvicorn api_model:app --host 0.0.0.0 --port 8000 --reload