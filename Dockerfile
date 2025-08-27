FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update
RUN apt-get install -y ffmpeg libsndfile1 build-essential git wget curl python3-pip pulseaudio libportaudio2 libpulse0 alsa-utils tzdata portaudio19-dev espeak
RUN rm -rf /var/lib/apt/lists/* 
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "Atlas.py"]