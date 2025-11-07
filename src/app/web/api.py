from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Optional

logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_instance = None
db_handler_instance = None


def set_instances(camera, db_handler):
    global camera_instance, db_handler_instance
    camera_instance = camera
    db_handler_instance = db_handler


@app.get("/")
async def root():
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "stream_raw": "/stream/raw",
            "stream_annotated": "/stream/annotated",
            "stats": "/api/stats",
            "persons": "/api/persons",
            "person": "/api/person/{person_id}"
        }
    }


@app.get("/api/stats")
async def get_statistics():
    if not db_handler_instance:
        raise HTTPException(status_code=503, detail="Database not available")
        
    stats = db_handler_instance.get_statistics()
    return JSONResponse(content=stats)


@app.get("/api/persons")
async def get_all_persons():
    if not db_handler_instance:
        raise HTTPException(status_code=503, detail="Database not available")
        
    persons = db_handler_instance.get_all_persons()
    
    result = []
    for person in persons:
        result.append({
            'person_id': person['person_id'],
            'first_seen': person['first_seen'].isoformat() if person.get('first_seen') else None,
            'last_seen': person['last_seen'].isoformat() if person.get('last_seen') else None,
            'detection_count': person.get('detection_count', 0)
        })
        
    return JSONResponse(content=result)


@app.get("/api/person/{person_id}")
async def get_person(person_id: str):
    if not db_handler_instance:
        raise HTTPException(status_code=503, detail="Database not available")
        
    person = db_handler_instance.get_person(person_id)
    
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
        
    return JSONResponse(content={
        'person_id': person['person_id'],
        'first_seen': person['first_seen'].isoformat() if person.get('first_seen') else None,
        'last_seen': person['last_seen'].isoformat() if person.get('last_seen') else None,
        'detection_count': person.get('detection_count', 0)
    })


@app.delete("/api/person/{person_id}")
async def delete_person(person_id: str):
    if not db_handler_instance:
        raise HTTPException(status_code=503, detail="Database not available")
        
    success = db_handler_instance.delete_person(person_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Person not found")
        
    return JSONResponse(content={"message": "Person deleted successfully"})


@app.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


async def start_api_server(config, camera, db_handler):
    set_instances(camera, db_handler)
    
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    from web.stream import setup_stream_routes
    setup_stream_routes(app, camera)
    
    config_obj = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config_obj)
    
    logger.info(f"Starting API server on {host}:{port}")
    await server.serve()