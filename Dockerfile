FROM python:3.12.7-slim-bookworm AS base
LABEL maintainer="Shenghao Wang <shenghao.wsh@gmail.com>"

WORKDIR /root

RUN pip install --upgrade pip
RUN apt update

##################################################### DEPS-NO-PIN ENVIRONMENT #####################################################
FROM base AS deps-no-pin

# Install linux packages required by repo
RUN apt -y install make git gcc g++ cmake

# Install core dependencies.
RUN apt-get update && apt-get install -y \
    libpq-dev build-essential libffi-dev libssl-dev python3-dev

# Remove apt binaries
RUN rm -rf /var/lib/apt/lists/*

# Set environment variable to avoid building wheels from source
ENV PIP_NO_BUILD_ISOLATION=1

COPY requirements-no-pin.txt .
RUN pip install --no-cache-dir -r requirements-no-pin.txt --prefer-binary

##################################################### DEPENDENCIES SETUP ##########################################################
FROM base AS final

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install linux packages required by repo
RUN apt -y install make git

# Remove apt binaries
RUN rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/root:./src"

# COPY the actual code
COPY . /root
