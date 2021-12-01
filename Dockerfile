FROM rocm/dev-ubuntu-20.04:4.3

WORKDIR /millipyde

RUN apt-get update

RUN apt-get install python3.8 -y

RUN apt install python3-pip -y

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

COPY . /millipyde

RUN /millipyde/build.sh

CMD ["python3"]