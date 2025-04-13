FROM python:3.9-slim-buster

WORKDIR /flask-docker

RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python","-m","flask","--app","Flask_app_predict.py","run","--host=0.0.0.0"]