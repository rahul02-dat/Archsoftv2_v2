from pymongo import MongoClient, ASCENDING
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseHandler:
    def __init__(self, config):
        self.config = config.get('database', {})
        
        mongo_uri = self.config.get('mongo_uri', 'mongodb://localhost:27017/')
        db_name = self.config.get('db_name', 'face_id')
        self.collection_name = self.config.get('collection_name', 'persons')
        
        logger.info(f"Connecting to MongoDB: {mongo_uri}")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[self.collection_name]
        
        self._create_indexes()
        logger.info("MongoDB connected successfully")
        
    def _create_indexes(self):
        self.collection.create_index([("person_id", ASCENDING)], unique=True)
        self.collection.create_index([("first_seen", ASCENDING)])
        self.collection.create_index([("last_seen", ASCENDING)])
        
    def insert_person(
        self,
        person_id: str,
        embedding: List[float],
        first_seen: datetime,
        last_seen: datetime
    ) -> bool:
        try:
            document = {
                'person_id': person_id,
                'embedding': embedding,
                'first_seen': first_seen,
                'last_seen': last_seen,
                'detection_count': 1
            }
            
            self.collection.insert_one(document)
            logger.info(f"Inserted person: {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting person: {e}")
            return False
            
    def update_person(self, person_id: str, last_seen: datetime) -> bool:
        try:
            result = self.collection.update_one(
                {'person_id': person_id},
                {
                    '$set': {'last_seen': last_seen},
                    '$inc': {'detection_count': 1}
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating person: {e}")
            return False
            
    def get_person(self, person_id: str) -> Optional[Dict]:
        try:
            return self.collection.find_one({'person_id': person_id})
        except Exception as e:
            logger.error(f"Error getting person: {e}")
            return None
            
    def get_all_persons(self) -> List[Dict]:
        try:
            return list(self.collection.find({}))
        except Exception as e:
            logger.error(f"Error getting all persons: {e}")
            return []
            
    def delete_person(self, person_id: str) -> bool:
        try:
            result = self.collection.delete_one({'person_id': person_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting person: {e}")
            return False
            
    def get_statistics(self) -> Dict:
        try:
            total_persons = self.collection.count_documents({})
            
            pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'total_detections': {'$sum': '$detection_count'}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            total_detections = result[0]['total_detections'] if result else 0
            
            recent_persons = list(
                self.collection.find({})
                .sort('last_seen', -1)
                .limit(10)
            )
            
            return {
                'total_persons': total_persons,
                'total_detections': total_detections,
                'recent_persons': [
                    {
                        'person_id': p['person_id'],
                        'last_seen': p['last_seen'].isoformat() if p.get('last_seen') else None,
                        'detection_count': p.get('detection_count', 0)
                    }
                    for p in recent_persons
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
            
    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed")