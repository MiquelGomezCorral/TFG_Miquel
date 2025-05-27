FROM ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV FLIT_ROOT_INSTALL=1 

RUN apt update -y --fix-missing \
&& apt upgrade -y \
&& apt install software-properties-common -y \
&& apt install pkg-config -y \
&& apt install python3-dev default-libmysqlclient-dev build-essential -y \
&& apt install sox ffmpeg libcairo2 libcairo2-dev -y \
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt update -y \
&& apt install python3.10 python3-pip poppler-utils poppler-utils git curl autoconf automake libtool -y \
&& apt install libgirepository1.0-dev -y \
&& apt install libicu-dev jq -y \
&& apt clean

# Set python to use python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set pip to use pip3
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory to /app
WORKDIR /app

# ====== Install python dependencies ======
RUN pip install --upgrade pip setuptools wheel tomlq

# Solver dependencies (private)
RUN pip install --extra-index-url https://solver:x6tDJ2to9Koz@pypi.solverml.com/ ocr-llm-module==0.0.1 

# ====== Set Root folder for imports & Install python dependencies ======
COPY ./app/setup.py ./
RUN pip install -e . 

COPY ./setup/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Expose port 8000 for the application if needed
# EXPOSE 8080