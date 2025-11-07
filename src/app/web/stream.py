from fastapi import Response
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import logging

logger = logging.getLogger(__name__)


def setup_stream_routes(app, camera):
    
    @app.get("/stream/raw")
    async def stream_raw():
        return StreamingResponse(
            generate_raw_stream(camera),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    @app.get("/stream/annotated")
    async def stream_annotated():
        return StreamingResponse(
            generate_annotated_stream(camera),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )


async def generate_raw_stream(camera):
    while True:
        frame = camera.get_latest_frame()
        
        if frame is None:
            await asyncio.sleep(0.03)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        await asyncio.sleep(0.03)


async def generate_annotated_stream(camera):
    while True:
        frame = camera.get_latest_annotated()
        
        if frame is None:
            frame = camera.get_latest_frame()
            
        if frame is None:
            await asyncio.sleep(0.03)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        await asyncio.sleep(0.03)