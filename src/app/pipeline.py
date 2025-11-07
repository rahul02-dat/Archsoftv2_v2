import cv2
import numpy as np
import logging
from typing import List, Tuple
import insightface
from insightface.app import FaceAnalysis
import asyncio

logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    def __init__(self, config, quality_checker, matcher, notifier):
        self.config = config.get('face_recognition', {})
        self.quality_checker = quality_checker
        self.matcher = matcher
        self.notifier = notifier
        
        self.det_thresh = self.config.get('det_thresh', 0.5)
        self.min_face_size = self.config.get('min_face_size', 40)
        
        logger.info("Initializing InsightFace models...")
        self.app = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_thresh=self.det_thresh, det_size=(640, 640))
        logger.info("InsightFace models loaded successfully")
        
    async def process_stream(self, camera):
        camera.start()
        
        frame_count = 0
        
        while True:
            frame = camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            frame_count += 1
            
            if frame_count % 2 == 0:
                await asyncio.sleep(0.01)
                continue
                
            try:
                annotated_frame = await self.process_frame(frame)
                camera.set_annotated_frame(annotated_frame)
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                camera.set_annotated_frame(frame)
                await asyncio.sleep(0.1)
            
    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        
        try:
            faces = self.app.get(frame)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return annotated
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            w = x2 - x1
            h = y2 - y1
            
            if w < self.min_face_size or h < self.min_face_size:
                continue
                
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                logger.warning(f"Invalid face crop: {face_img.shape}")
                continue
            
            quality_score, quality_issues = self.quality_checker.check_quality(
                face_img, face
            )
            
            if quality_score < 0.5:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated, "Low Quality", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
                continue
                
            embedding = face.normed_embedding
            
            match_result = await self.matcher.match_face(embedding)
            
            if match_result['matched']:
                person_id = match_result['person_id']
                confidence = match_result['confidence']
                
                color = (0, 255, 0)
                label = f"ID: {person_id} ({confidence:.2f})"
                
                if match_result['is_new_detection']:
                    await self.notifier.notify(
                        person_id=person_id,
                        confidence=confidence,
                        bbox=bbox
                    )
            else:
                person_id = match_result['person_id']
                color = (255, 0, 0)
                label = f"New: {person_id}"
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            
            timestamp = match_result.get('last_seen', '')
            if timestamp:
                cv2.putText(
                    annotated, timestamp, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
                
        return annotated