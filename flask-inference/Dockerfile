FROM amazonlinux:2

RUN yum update -y
RUN yum install git -y
RUN yum install python3-pip -y
RUN pip3 install --upgrade pip

WORKDIR /home
RUN git clone https://github.com/ddps-lab/ai-platform.git
WORKDIR ai-platform/flask-inference/

RUN mkdir -p static/uploads

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]
