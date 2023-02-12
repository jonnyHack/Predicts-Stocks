FROM python:3.9

EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt




COPY . .



#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
#CMD ["streamlit", "run" ,"main.py", "--host", "0.0.0.0", "--port", "8081"]
#ENTRYPOINT ["streamlit", "run", "main.py:app", "--server.port=8081", "--server.address=0.0.0.0"]
CMD ["streamlit" ,"run", "main.py"]



