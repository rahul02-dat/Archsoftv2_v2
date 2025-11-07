import numpy as np
import logging
from typing import Dict, Optional
from datetime import datetime
from utils.id_generator import generate_unique_id

logger = logging.getLogger(__name__)


class FaceMatcher:
    def __init__(self, config, db_handler):
        self.config = config.get('matching', {})
        self.db_handler = db_handler
        
        self.threshold = self.config.get('threshold', 0.6)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
    async def match_face(self, embedding: np.ndarray) -> Dict:
        all_persons = self.db_handler.get_all_persons()
        
        best_match = None
        best_similarity = -1
        
        for person in all_persons:
            stored_embedding = np.array(person['embedding'])
            similarity = self.compute_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person
                
        current_time = datetime.utcnow()
        
        if best_similarity >= self.threshold:
            person_id = best_match['person_id']
            
            last_seen_dt = best_match.get('last_seen')
            is_new_detection = True
            
            if last_seen_dt:
                time_diff = (current_time - last_seen_dt).total_seconds()
                if time_diff < 60:
                    is_new_detection = False
                    
            self.db_handler.update_person(
                person_id=person_id,
                last_seen=current_time
            )
            
            return {
                'matched': True,
                'person_id': person_id,
                'confidence': float(best_similarity),
                'is_new_detection': is_new_detection,
                'last_seen': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            person_id = generate_unique_id()
            
            self.db_handler.insert_person(
                person_id=person_id,
                embedding=embedding.tolist(),
                first_seen=current_time,
                last_seen=current_time
            )
            
            logger.info(f"New person registered: {person_id}")
            
            return {
                'matched': False,
                'person_id': person_id,
                'confidence': 0.0,
                'is_new_detection': True,
                'last_seen': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }