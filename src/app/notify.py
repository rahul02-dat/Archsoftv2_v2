import logging
from typing import Optional
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config):
        self.config = config.get('notifications', {})
        
        self.enabled = self.config.get('enabled', True)
        self.desktop_enabled = self.config.get('desktop', False)
        self.websocket_enabled = self.config.get('websocket', True)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 60)
        
        self.last_notification = defaultdict(lambda: datetime.min)
        self.websocket_clients = set()
        
    def should_notify(self, person_id: str) -> bool:
        if not self.enabled:
            return False
            
        last_time = self.last_notification[person_id]
        current_time = datetime.utcnow()
        
        if current_time - last_time < timedelta(seconds=self.cooldown_seconds):
            return False
            
        return True
        
    async def notify(
        self,
        person_id: str,
        confidence: float,
        bbox: Optional[list] = None
    ):
        if not self.should_notify(person_id):
            return
            
        self.last_notification[person_id] = datetime.utcnow()
        
        message = f"Person {person_id} detected (confidence: {confidence:.2f})"
        
        if self.desktop_enabled:
            await self._send_desktop_notification(message)
            
        if self.websocket_enabled:
            await self._send_websocket_notification({
                'type': 'face_detected',
                'person_id': person_id,
                'confidence': confidence,
                'bbox': bbox,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        logger.info(f"Notification sent: {message}")
        
    async def _send_desktop_notification(self, message: str):
        try:
            from plyer import notification
            notification.notify(
                title="Face Recognition Alert",
                message=message,
                timeout=5
            )
        except Exception as e:
            logger.error(f"Desktop notification failed: {e}")
            
    async def _send_websocket_notification(self, data: dict):
        if not self.websocket_clients:
            return
            
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                await client.send_json(data)
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")
                disconnected.add(client)
                
        self.websocket_clients -= disconnected
        
    def add_websocket_client(self, client):
        self.websocket_clients.add(client)
        logger.info(f"WebSocket client added. Total: {len(self.websocket_clients)}")
        
    def remove_websocket_client(self, client):
        self.websocket_clients.discard(client)
        logger.info(f"WebSocket client removed. Total: {len(self.websocket_clients)}")