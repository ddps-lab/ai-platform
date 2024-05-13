FROM ubuntu:22.04

RUN apt update -y
RUN apt install git -y
RUN apt install python3-pip -y
RUN pip3 install --upgrade pip

WORKDIR /home
RUN git clone https://github.com/ddps-lab/ai-platform.git
WORKDIR ai-platform/flask-inference/
COPY flower_model.keras .

RUN mkdir -p static/uploads

RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["python3", "main.py"]
