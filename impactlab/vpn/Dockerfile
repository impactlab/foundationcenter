# Pull base image.
FROM ubuntu

# Install curl
RUN \
  apt-get update && \
  apt-get install -y tar curl && \ 
  rm -rf /var/lib/apt/lists/*

# Install netExtender VPN
RUN \ 
  curl -O https://sslvpn.demo.sonicwall.com/NetExtender.Linux.7.5.771.x86_64.tgz && \
  tar zvxf NetExtender.Linux.7.5.771.x86_64.tgz && \ 
  ln -s /lib/x86_64-linux-gnu/libssl.so.1.0.0 /usr/lib/libssl.so.6 && \ 
  ln -s /lib/x86_64-linux-gnu/libcrypto.so.1.0.0 /usr/lib/libcrypto.so.6 && \ 
  
# Define working directory.
WORKDIR /netExtenderClient

CMD netExtender -u $VPN_USERNAME -p $VPN_PASSWORD -d $VPN_SERVER