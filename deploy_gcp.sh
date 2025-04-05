#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-venv git

# Create app directory
mkdir -p /opt/trading-agent
cd /opt/trading-agent

# Clone repository (replace with your actual repository URL)
# git clone https://github.com/yourusername/trading-agent.git .
# Or copy files manually if not using git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p data

# Create systemd service
sudo tee /etc/systemd/system/trading-agent.service << EOF
[Unit]
Description=Trading Agent Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/trading-agent
Environment=PYTHONPATH=/opt/trading-agent
ExecStart=/opt/trading-agent/venv/bin/python main.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-agent
sudo systemctl start trading-agent

# Setup log rotation
sudo tee /etc/logrotate.d/trading-agent << EOF
/opt/trading-agent/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $USER $USER
}
EOF

# Setup Google Cloud Logging
sudo tee /opt/trading-agent/utils/gcp_logging.py << EOF
"""Google Cloud Logging integration."""
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

def setup_gcp_logging(logger_name="trading-agent"):
    """Setup Google Cloud Logging."""
    try:
        client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(client, name=logger_name)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Get the root logger and add the handler
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        return True
    except Exception as e:
        print(f"Failed to setup GCP logging: {str(e)}")
        return False
EOF

# Update main.py to use GCP logging if available
if [ -f "/opt/trading-agent/main.py" ]; then
    sed -i 's/from logger import setup_logger/from logger import setup_logger\nfrom utils.gcp_logging import setup_gcp_logging/g' /opt/trading-agent/main.py
    sed -i 's/logger = setup_logger()/logger = setup_logger()\ntry:\n    setup_gcp_logging()\nexcept Exception as e:\n    logger.error(f"Failed to setup GCP logging: {str(e)}")/g' /opt/trading-agent/main.py
fi

echo "Deployment completed successfully!" 