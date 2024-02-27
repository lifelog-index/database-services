# uvicorn main:app --reload --port 7777 --host 0.0.0.0
uvicorn main:app --port 20541 --host 0.0.0.0 --workers 1 # limit create models
