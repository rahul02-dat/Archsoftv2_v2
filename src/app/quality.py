import cv2
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class QualityChecker:
    def __init__(self, config):
        self.config = config.get('quality', {})
        
        self.min_blur_threshold = self.config.get('min_blur_threshold', 150)
        self.min_brightness = self.config.get('min_brightness', 50)
        self.max_brightness = self.config.get('max_brightness', 210)
        self.min_face_size = self.config.get('min_face_size', 60)
        
        logger.info(f"QualityChecker initialized: blur_thresh={self.min_blur_threshold}, "
                   f"brightness=[{self.min_brightness}, {self.max_brightness}], "
                   f"min_size={self.min_face_size}")
        
    def check_blur(self, face_img: np.ndarray) -> Tuple[float, bool]:
        """Check if face image is blurry using Laplacian variance"""
        if face_img is None or face_img.size == 0:
            return 0.0, False
            
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            is_sharp = laplacian_var >= self.min_blur_threshold
            
            logger.debug(f"Blur check: variance={laplacian_var:.1f}, sharp={is_sharp}")
            
            return laplacian_var, is_sharp
        except Exception as e:
            logger.error(f"Blur check failed: {e}")
            return 0.0, False
        
    def check_brightness(self, face_img: np.ndarray) -> Tuple[float, bool]:
        """Check if face has good lighting/brightness"""
        if face_img is None or face_img.size == 0:
            return 0.0, False
            
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            is_good = self.min_brightness <= mean_brightness <= self.max_brightness
            
            logger.debug(f"Brightness check: value={mean_brightness:.1f}, good={is_good}")
            
            return mean_brightness, is_good
        except Exception as e:
            logger.error(f"Brightness check failed: {e}")
            return 0.0, False
        
    def check_face_size(self, face) -> Tuple[int, bool]:
        """Check if face is large enough for reliable recognition"""
        bbox = face.bbox.astype(int)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        min_dim = min(width, height)
        is_large_enough = min_dim >= self.min_face_size
        
        logger.debug(f"Size check: {width}x{height} (min={min_dim}), large_enough={is_large_enough}")
        
        return min_dim, is_large_enough
        
    def check_pose(self, face) -> Tuple[float, bool]:
        """Check if face has frontal pose (not too much rotation)"""
        if not hasattr(face, 'pose') or face.pose is None:
            logger.debug("No pose information available, assuming frontal")
            return 0.0, True
            
        try:
            pose = face.pose
            
            pitch = abs(pose[0]) if len(pose) > 0 else 0
            yaw = abs(pose[1]) if len(pose) > 1 else 0
            roll = abs(pose[2]) if len(pose) > 2 else 0
            
            max_angle = max(pitch, yaw, roll)
            
            # More strict threshold for frontal faces
            is_frontal = max_angle < 25
            
            logger.debug(f"Pose check: pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f}, frontal={is_frontal}")
            
            return max_angle, is_frontal
        except Exception as e:
            logger.error(f"Pose check failed: {e}")
            return 0.0, True
        
    def check_quality(
        self,
        face_img: np.ndarray,
        face
    ) -> Tuple[float, List[str]]:
        """
        Comprehensive quality check for face image
        Returns: (overall_score, list_of_issues)
        """
        issues = []
        scores = []
        
        # Check blur
        blur_score, is_sharp = self.check_blur(face_img)
        if not is_sharp:
            issues.append("blurry")
            scores.append(0.0)
        else:
            # Normalize blur score (higher is better)
            normalized_blur = min(1.0, blur_score / 300)
            scores.append(normalized_blur)
            
        # Check brightness
        brightness, is_good_light = self.check_brightness(face_img)
        if not is_good_light:
            if brightness < self.min_brightness:
                issues.append("too_dark")
            else:
                issues.append("too_bright")
            scores.append(0.0)
        else:
            # Normalize brightness (closer to 128 is better)
            normalized_brightness = 1.0 - (abs(brightness - 128) / 128)
            scores.append(normalized_brightness)
            
        # Check face size
        face_size, is_large = self.check_face_size(face)
        if not is_large:
            issues.append("too_small")
            scores.append(0.0)
        else:
            # Normalize size score
            normalized_size = min(1.0, face_size / 150)
            scores.append(normalized_size)
            
        # Check pose
        pose_angle, is_frontal = self.check_pose(face)
        if not is_frontal:
            issues.append("bad_pose")
            scores.append(0.0)
        else:
            # Normalize pose score (lower angle is better)
            normalized_pose = 1.0 - (pose_angle / 25)
            scores.append(normalized_pose)
            
        # Calculate overall quality score
        if not scores:
            overall_score = 0.0
        else:
            overall_score = np.mean(scores)
        
        logger.debug(f"Quality check result: score={overall_score:.2f}, issues={issues}")
        
        return overall_score, issues