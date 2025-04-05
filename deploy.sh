#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-venv git

# Create app directory
mkdir -p /opt/trading-agent
cd /opt/trading-agent

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
User=ubuntu
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
    create 0640 ubuntu ubuntu
}
EOF

echo "Deployment completed successfully!" 