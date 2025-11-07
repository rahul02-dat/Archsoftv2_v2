import cv2
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class QualityChecker:
    def __init__(self, config):
        self.config = config.get('quality', {})
        
        self.min_blur_threshold = self.config.get('min_blur_threshold', 100)
        self.min_brightness = self.config.get('min_brightness', 40)
        self.max_brightness = self.config.get('max_brightness', 220)
        self.min_face_size = self.config.get('min_face_size', 40)
        
    def check_blur(self, face_img: np.ndarray) -> Tuple[float, bool]:
        if face_img is None or face_img.size == 0:
            return 0.0, False
            
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            is_sharp = laplacian_var >= self.min_blur_threshold
            
            return laplacian_var, is_sharp
        except Exception as e:
            logger.error(f"Blur check failed: {e}")
            return 0.0, False
        
    def check_brightness(self, face_img: np.ndarray) -> Tuple[float, bool]:
        if face_img is None or face_img.size == 0:
            return 0.0, False
            
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            is_good = self.min_brightness <= mean_brightness <= self.max_brightness
            
            return mean_brightness, is_good
        except Exception as e:
            logger.error(f"Brightness check failed: {e}")
            return 0.0, False
        
    def check_face_size(self, face) -> Tuple[int, bool]:
        bbox = face.bbox.astype(int)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        min_dim = min(width, height)
        is_large_enough = min_dim >= self.min_face_size
        
        return min_dim, is_large_enough
        
    def check_pose(self, face) -> Tuple[float, bool]:
        if not hasattr(face, 'pose'):
            return 0.0, True
            
        pose = face.pose
        
        pitch = abs(pose[0]) if len(pose) > 0 else 0
        yaw = abs(pose[1]) if len(pose) > 1 else 0
        roll = abs(pose[2]) if len(pose) > 2 else 0
        
        max_angle = max(pitch, yaw, roll)
        
        is_frontal = max_angle < 30
        
        return max_angle, is_frontal
        
    def check_quality(
        self,
        face_img: np.ndarray,
        face
    ) -> Tuple[float, List[str]]:
        issues = []
        scores = []
        
        blur_score, is_sharp = self.check_blur(face_img)
        if not is_sharp:
            issues.append("blurry")
            scores.append(0.0)
        else:
            scores.append(min(1.0, blur_score / 200))
            
        brightness, is_good_light = self.check_brightness(face_img)
        if not is_good_light:
            issues.append("poor_lighting")
            scores.append(0.0)
        else:
            normalized_brightness = abs(brightness - 128) / 128
            scores.append(1.0 - normalized_brightness)
            
        face_size, is_large = self.check_face_size(face)
        if not is_large:
            issues.append("too_small")
            scores.append(0.0)
        else:
            scores.append(min(1.0, face_size / 100))
            
        pose_angle, is_frontal = self.check_pose(face)
        if not is_frontal:
            issues.append("bad_pose")
            scores.append(0.0)
        else:
            scores.append(1.0 - (pose_angle / 30))
            
        overall_score = np.mean(scores) if scores else 0.0
        
        return overall_score, issues