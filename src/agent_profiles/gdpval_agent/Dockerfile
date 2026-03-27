FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libreoffice --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*