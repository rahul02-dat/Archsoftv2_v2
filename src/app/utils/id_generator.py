import uuid
import hashlib
from datetime import datetime


def generate_unique_id() -> str:
    timestamp = datetime.utcnow().isoformat()
    random_uuid = str(uuid.uuid4())
    
    combined = f"{timestamp}_{random_uuid}"
    
    hash_object = hashlib.sha256(combined.encode())
    hash_hex = hash_object.hexdigest()
    
    short_id = hash_hex[:12].upper()
    
    return f"PERSON_{short_id}"


def generate_session_id() -> str:
    return str(uuid.uuid4())