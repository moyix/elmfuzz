# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install Python dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    gpg-agent \
    python3.11-venv \
    python3-pip \
    python3.11-distutils \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*
    # add-apt-repository ppa:deadsnakes/ppa && \
    # apt-get update && apt-get install -y python3.11 \
    # --no-install-recommends && \

# Create a non-root user
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Set up a virtual environment for the non-root user
RUN python3.11 -m venv venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Upgrade pip and install wheel within the virtual environment
RUN pip install --upgrade pip wheel

# Make the app directory
RUN mkdir app

# Your app's setup could go here, for example:
COPY --chown=appuser:appuser ../driver.py ../drive_log.py /home/appuser/app/
WORKDIR /home/appuser/app
