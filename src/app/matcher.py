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
        
        logger.info(f"FaceMatcher initialized with threshold={self.threshold}")
        
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-6)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
        
    async def match_face(self, embedding: np.ndarray) -> Dict:
        """Match a face embedding against the database"""
        
        # Validate embedding
        if embedding is None or len(embedding) == 0:
            logger.error("Invalid embedding provided")
            return {
                'matched': False,
                'person_id': None,
                'confidence': 0.0,
                'is_new_detection': False,
                'error': 'Invalid embedding'
            }
        
        # Get all persons from database
        all_persons = self.db_handler.get_all_persons()
        
        if not all_persons:
            logger.info("Database is empty, registering first person")
            return await self._register_new_person(embedding)
        
        logger.debug(f"Comparing against {len(all_persons)} persons in database")
        
        best_match = None
        best_similarity = -1
        
        # Find best match
        for person in all_persons:
            try:
                stored_embedding = np.array(person['embedding'])
                
                # Validate stored embedding
                if stored_embedding is None or len(stored_embedding) == 0:
                    logger.warning(f"Invalid stored embedding for person {person['person_id']}")
                    continue
                
                similarity = self.compute_similarity(embedding, stored_embedding)
                
                logger.debug(f"Similarity with {person['person_id']}: {similarity:.3f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person
                    
            except Exception as e:
                logger.error(f"Error comparing with person {person['person_id']}: {e}")
                continue
        
        current_time = datetime.utcnow()
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.threshold:
            logger.info(f"Match found: {best_match['person_id']} with similarity {best_similarity:.3f}")
            
            person_id = best_match['person_id']
            
            # Check if this is a new detection (based on cooldown)
            last_seen_dt = best_match.get('last_seen')
            is_new_detection = True
            
            if last_seen_dt:
                time_diff = (current_time - last_seen_dt).total_seconds()
                if time_diff < 60:  # Within cooldown period
                    is_new_detection = False
                    logger.debug(f"Within cooldown period ({time_diff:.1f}s)")
            
            # Update person in database
            self.db_handler.update_person(
                person_id=person_id,
                last_seen=current_time
            )
            
            # Get updated person data
            updated_person = self.db_handler.get_person(person_id)
            
            return {
                'matched': True,
                'person_id': person_id,
                'confidence': float(best_similarity),
                'is_new_detection': is_new_detection,
                'first_seen': updated_person['first_seen'].strftime('%Y-%m-%d %H:%M:%S') if updated_person.get('first_seen') else 'N/A',
                'last_seen': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'detection_count': updated_person.get('detection_count', 0)
            }
        else:
            # No match found, register as new person
            logger.info(f"No match found (best similarity: {best_similarity:.3f}), registering new person")
            return await self._register_new_person(embedding)
    
    async def _register_new_person(self, embedding: np.ndarray) -> Dict:
        """Register a new person in the database"""
        person_id = generate_unique_id()
        current_time = datetime.utcnow()
        
        success = self.db_handler.insert_person(
            person_id=person_id,
            embedding=embedding.tolist(),
            first_seen=current_time,
            last_seen=current_time
        )
        
        if not success:
            logger.error(f"Failed to register new person {person_id}")
            return {
                'matched': False,
                'person_id': None,
                'confidence': 0.0,
                'is_new_detection': False,
                'error': 'Database insertion failed'
            }
        
        logger.info(f"New person registered: {person_id}")
        
        return {
            'matched': False,
            'person_id': person_id,
            'confidence': 0.0,
            'is_new_detection': True,
            'first_seen': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_seen': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'detection_count': 1
        }