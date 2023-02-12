FROM python:3.9

#RUN pip install fastapi
#RUN pip install "uvicorn[standard]"
#RUN pip install numpy
#RUN pip install matplotlib
#RUN pip install pandas 
#RUN pip install pandas-datareader
##RUN pip install tensorflow
##RUN pip install scikit-learn
#RUN pip install pytest

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
