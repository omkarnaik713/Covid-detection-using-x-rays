FROM python 3.11 
COPY .
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000 
CMD python app.py