from ubuntu:20.04

run apt update && apt install -y python3 python3-pip

# git
run apt install -y git
run git clone https://github.com/openai/point-e

# install
run pip3 install -e point-e

run pip install \
torch \
tqdm \
streamlit
