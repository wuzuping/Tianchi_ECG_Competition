FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3
ADD . /competition
WORKDIR /competition
RUN pip --no-cache-dir install  -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN sudo chmod -R 777 /competition
CMD ['sh', 'run.sh']
