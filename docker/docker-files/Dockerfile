FROM pytorch/pytorch:latest

LABEL maintainer="Nils Reimers <info@nils-reimers>"

RUN apt-get update && apt-get -y install build-essential

### Copied from: https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.8.dockerfile
RUN pip install --no-cache-dir "uvicorn[standard]" gunicorn fastapi

COPY ./start_backend.sh /start_backend.sh
RUN chmod +x /start_backend.sh

COPY ./start_frontend.sh /start_frontend.sh
RUN chmod +x /start_frontend.sh

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./gunicorn_conf_backend.py /gunicorn_conf_backend.py
COPY ./gunicorn_conf_frontend.py /gunicorn_conf_frontend.py

#COPY ./start-reload.sh /start-reload.sh
#RUN chmod +x /start-reload.sh




#####
COPY ../api/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

####

COPY ../api /app
WORKDIR /app/
ENV PYTHONPATH=/app
EXPOSE 80

####

# Run the start script
CMD ["/start.sh"]



