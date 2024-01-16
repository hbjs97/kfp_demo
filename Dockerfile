# Python 3.9 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir torch torchvision numpy boto3 kfp==1.8.9 scikit-learn==1.0.1 mlflow==1.21.0 pandas==1.3.4 dill==0.3.4

# Docker 컨테이너가 실행될 때 기본적으로 수행할 명령을 설정
# 여기서는 bash 쉘을 기본 명령으로 설정하였습니다.
CMD ["/bin/bash"]
