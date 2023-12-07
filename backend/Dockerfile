FROM pytorch/pytorch:latest
RUN pip install --upgrade pip

WORKDIR /usr/src/rag_chat
COPY . .
RUN apt-get -y update && apt-get -y install build-essential
RUN cd ./builds/llama.cpp && make


WORKDIR /usr/src/rag_chat
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
CMD ["python", "__init__.py"]