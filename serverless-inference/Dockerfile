# lambda base image
FROM amazon/aws-lambda-python:3.8

RUN yum update -y
RUN yum install git -y
RUN yum install python3-pip -y
RUN pip3 install --upgrade pip

RUN git clone https://github.com/ddps-lab/architect-cloud.git
WORKDIR architect-cloud/sample-application/serverless-inference/

RUN mkdir -p static/uploads

RUN pip3 install -r requirements.txt

# Move lambda_function.py from the Git repository to /var/task/ inside the container
RUN cp lambda_function_container.py /var/task/

# RUN lambda_function_container.handler
CMD ["lambda_function_container.handler"]
