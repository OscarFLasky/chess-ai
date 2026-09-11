from fastapi import FastAPI

app = FastAPI(title = "test API", version = "0.1.0")

@app.get("/")
def racine():
    return {"status":"ok"}

@app.get("/version")
def version():
    return {"version": app.version}

