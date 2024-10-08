FROM python
RUN pip install --upgrade pip 
RUN apt-get update && apt-get install -y libhdf5-dev
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD python app.py