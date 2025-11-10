"""
Entry point for defender service
"""
from .api import app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Defender Malware Detection Service on port 8080")
    app.run(host='0.0.0.0', port=8080, debug=False)