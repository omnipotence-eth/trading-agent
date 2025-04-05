#!/bin/bash

# Backup script for trading agent data
BACKUP_DIR="/opt/trading-agent/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.tar.gz"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create backup
tar -czf $BACKUP_FILE /opt/trading-agent/data /opt/trading-agent/logs /opt/trading-agent/.env

# Keep only the last 7 backups
ls -t $BACKUP_DIR/backup_*.tar.gz | tail -n +8 | xargs -r rm

# Upload to Google Cloud Storage (optional)
# gsutil cp $BACKUP_FILE gs://your-bucket-name/trading-agent-backups/

echo "Backup completed: $BACKUP_FILE" 