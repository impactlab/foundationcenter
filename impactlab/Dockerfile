# Pull base image.
FROM ubuntu

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y python python-dev python-pip python-virtualenv \
  python-numpy python-scipy python-pandas python-nose && \
  rm -rf /var/lib/apt/lists/*

# Install packages. 
RUN \ 
  pip install Cython && \  
  pip install -r requirements.txt

EXPOSE 9090 

# Define working directory.
WORKDIR /

# Define default command.
CMD python grantServer.py