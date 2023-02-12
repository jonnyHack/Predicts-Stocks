# hello-world


```bash
 docker build -t fastapi_demo .
 docker run -ti -p8989:8080 fastapi_demo
```


```bash
 curl -X GET localhost:8989/aaaa\?a='2**2'
 curl -X GET localhost:8989/aaaa\?a='sys.exit(1)'
```
