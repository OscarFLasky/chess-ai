from fastapi import HTTPException, Request

def get_engine(request: Request):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not loaded")
    return engine
