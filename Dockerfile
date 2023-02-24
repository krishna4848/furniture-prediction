
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9 

WORKDIR /server

COPY requirements.txt  requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py main.py 
COPY my_model.h5 my_model.h5

COPY run.sh run.sh
RUN chmod +x run.sh
ENTRYPOINT "/server/run.sh"

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
