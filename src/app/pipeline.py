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
            
            # Process every 3rd frame to reduce load
            if frame_count % 3 != 0:
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
            # Detect faces in the frame
            faces = self.app.get(frame)
            
            # If no faces detected, return the original frame
            if not faces or len(faces) == 0:
                logger.debug("No faces detected in frame")
                return annotated
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return annotated
        
        logger.info(f"Detected {len(faces)} face(s) in frame")
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            w = x2 - x1
            h = y2 - y1
            
            # Check minimum face size
            if w < self.min_face_size or h < self.min_face_size:
                logger.debug(f"Face too small: {w}x{h}")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(
                    annotated, "Too Small", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2
                )
                continue
                
            # Extract face region
            face_img = frame[y1:y2, x1:x2]
            
            # Validate face crop
            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                logger.warning(f"Invalid face crop: {face_img.shape}")
                continue
            
            # Check face quality
            quality_score, quality_issues = self.quality_checker.check_quality(
                face_img, face
            )
            
            logger.debug(f"Face quality score: {quality_score:.2f}, issues: {quality_issues}")
            
            # Reject low quality faces
            if quality_score < 0.5:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                issue_text = ", ".join(quality_issues) if quality_issues else "Low Quality"
                cv2.putText(
                    annotated, issue_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
                logger.debug(f"Face rejected due to quality issues: {quality_issues}")
                continue
            
            # Get face embedding - THIS IS THE KEY PART
            if not hasattr(face, 'normed_embedding') or face.normed_embedding is None:
                logger.warning("Face has no embedding, skipping")
                continue
                
            embedding = face.normed_embedding
            
            # Verify embedding is valid
            if embedding is None or len(embedding) == 0:
                logger.warning("Invalid embedding, skipping face")
                continue
            
            logger.info(f"Processing face with embedding shape: {embedding.shape}")
            
            # Match face against database
            match_result = await self.matcher.match_face(embedding)
            
            if match_result['matched']:
                person_id = match_result['person_id']
                confidence = match_result['confidence']
                first_seen = match_result.get('first_seen', 'N/A')
                last_seen = match_result.get('last_seen', 'N/A')
                detection_count = match_result.get('detection_count', 0)
                
                color = (0, 255, 0)  # Green for known faces
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Display person ID and confidence
                label = f"ID: {person_id}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                
                # Display confidence
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(
                    annotated, conf_text, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                
                # Display detection count
                count_text = f"Seen: {detection_count}x"
                cv2.putText(
                    annotated, count_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
                
                # Display first seen
                first_text = f"First: {first_seen}"
                cv2.putText(
                    annotated, first_text, (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
                
                # Display last seen
                last_text = f"Last: {last_seen}"
                cv2.putText(
                    annotated, last_text, (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
                
                logger.info(f"Matched person: {person_id} with confidence {confidence:.2f}")
                
                # Send notification for new detection
                if match_result['is_new_detection']:
                    await self.notifier.notify(
                        person_id=person_id,
                        confidence=confidence,
                        bbox=bbox.tolist()
                    )
            else:
                # New person detected
                person_id = match_result['person_id']
                first_seen = match_result.get('first_seen', 'N/A')
                
                color = (255, 165, 0)  # Orange for new faces
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Display new person label
                label = f"NEW: {person_id}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                
                # Display first seen
                first_text = f"First: {first_seen}"
                cv2.putText(
                    annotated, first_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
                
                logger.info(f"New person registered: {person_id}")
                
                # Send notification for new person
                await self.notifier.notify(
                    person_id=person_id,
                    confidence=0.0,
                    bbox=bbox.tolist()
                )
                
        return annotated