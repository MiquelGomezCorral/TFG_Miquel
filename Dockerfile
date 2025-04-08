FROM ubuntu:22.04
 
RUN apt update -y --fix-missing
RUN apt upgrade -y
RUN apt install software-properties-common -y
RUN apt install pkg-config -y
RUN apt install python3-dev default-libmysqlclient-dev build-essential -y
RUN apt install sox ffmpeg libcairo2 libcairo2-dev -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update -y
RUN apt install python3.10 python3-pip poppler-utils poppler-utils git curl autoconf automake libtool -y
RUN apt install libgirepository1.0-dev -y


# Set python to use python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set pip to use pip3
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install python dependencies
RUN apt install libicu-dev jq -y

# Clean up the apt cache to reduce the image size
RUN apt clean


ENV PYTHONDONTWRITEBYTECODE=1
ENV FLIT_ROOT_INSTALL=1 

# Set the working directory to /app
WORKDIR /app

# ====== Install python dependencies ======
RUN pip install --upgrade pip setuptools wheel tomlq

# Solver dependencies (privatte)
RUN pip install --extra-index-url https://solver:x6tDJ2to9Koz@pypi.solverml.com/ ocr-llm-module==0.0.1 


# Custom dependencies
COPY ./scripts_environment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Expose port 8000 for the application
# EXPOSE 8080