import asyncio
import logging
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from capture import CameraCapture
from pipeline import FaceRecognitionPipeline
from matcher import FaceMatcher
from db_handler import DatabaseHandler
from quality import QualityChecker
from notify import Notifier
from utils.config_loader import ConfigLoader
from web.api import start_api_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceIDSystem:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = ConfigLoader(config_path)
        
        self.db_handler = DatabaseHandler(self.config)
        self.quality_checker = QualityChecker(self.config)
        self.matcher = FaceMatcher(self.config, self.db_handler)
        self.notifier = Notifier(self.config)
        
        self.camera = CameraCapture(self.config)
        self.pipeline = FaceRecognitionPipeline(
            self.config,
            self.quality_checker,
            self.matcher,
            self.notifier
        )
        
    async def run(self):
        logger.info("Starting Face ID System...")
        
        try:
            api_task = asyncio.create_task(
                start_api_server(self.config, self.camera, self.db_handler)
            )
            
            processing_task = asyncio.create_task(
                self.pipeline.process_stream(self.camera)
            )
            
            await asyncio.gather(api_task, processing_task)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
        finally:
            logger.info("Cleaning up resources...")
            self.camera.release()
            self.db_handler.close()
            logger.info("Shutdown complete")


if __name__ == "__main__":
    system = FaceIDSystem()
    asyncio.run(system.run())