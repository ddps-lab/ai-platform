FROM amazon/aws-lambda-python:3.9

RUN yum update -y && \
    yum install git -y && \
    yum install python3-pip -y && \
    pip3 install --upgrade pip

RUN git clone https://github.com/ddps-lab/ai-platform.git
WORKDIR ai-platform/serverless-inference/lambda-image-build/

#Download the model from the S3 bucket below:
#s3://ai-platform-flower-model/flower_model.keras .
COPY flower_model.keras .

RUN mkdir -p static/uploads
RUN pip3 install -r requirements.txt --no-cache-dir
RUN cp lambda_function_container.py /var/task

CMD ["lambda_function_container.handler"]
