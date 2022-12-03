FROM gcc:10-buster
COPY blitz /home/blitz
RUN echo "deb-src http://deb.debian.org/debian buster main"  >> /etc/apt/sources.list
RUN apt update -y
RUN apt upgrade -y
RUN apt install python3
ENTRYPOINT [ "python3", "/home/blitz/blitz.py" ]
# $docker build -t blitz .
# $docker run -v /hdd/data_c:/datac --security-opt seccomp=unconfined -it dkr:latest
