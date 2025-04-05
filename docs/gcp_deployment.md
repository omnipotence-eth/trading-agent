# Deploying the Trading Agent on Google Cloud Platform

This guide walks you through deploying the trading agent on Google Cloud Platform (GCP) using a free-tier eligible VM instance.

## Prerequisites

- A Google Cloud Platform account
- The Google Cloud SDK installed on your local machine
- Your trading agent code in a GitHub repository

## Step 1: Create a GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project"
4. Name it "trading-agent" (or your preferred name)
5. Click "Create"

## Step 2: Enable Required APIs

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for and enable the following APIs:
   - Compute Engine API
   - Cloud Logging API
   - Cloud Monitoring API

## Step 3: Create a VM Instance

1. In the Google Cloud Console, go to "Compute Engine" > "VM instances"
2. Click "Create Instance"
3. Configure the instance:
   - Name: `trading-agent`
   - Region: Choose a region close to your target market (e.g., `us-east1` for US East)
   - Machine configuration: 
     - Series: E2
     - Machine type: e2-micro (free tier eligible)
   - Boot disk:
     - Operating system: Ubuntu
     - Version: Ubuntu 20.04 LTS
     - Size: 10 GB (standard)
   - Firewall:
     - Allow HTTP traffic: No
     - Allow HTTPS traffic: No
   - Click "Create"

## Step 4: Set Up SSH Access

1. After the VM is created, click on the "SSH" button next to your instance
2. This will open a browser-based SSH terminal to your VM

## Step 5: Deploy Your Application

1. In the SSH terminal, create a directory for your application:
   ```bash
   mkdir -p ~/trading-agent
   cd ~/trading-agent
   ```

2. Clone your repository:
   ```bash
   git clone https://github.com/yourusername/trading-agent.git .
   ```

3. Make the deployment script executable:
   ```bash
   chmod +x ~/trading-agent/deploy_gcp.sh
   ```

4. Run the deployment script:
   ```bash
   cd ~/trading-agent
   ./deploy_gcp.sh
   ```

## Step 6: Set Up Environment Variables

1. Create a `.env` file on the VM:
   ```bash
   nano ~/trading-agent/.env
   ```

2. Add your API keys and credentials:
   ```
   X_API_KEY=your_twitter_api_key
   X_API_SECRET=your_twitter_api_secret
   X_ACCESS_TOKEN=your_twitter_access_token
   X_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   GROK_API_KEY=your_grok_api_key
   NEWSAPI_KEY=your_newsapi_key
   FINNHUB_API_KEY=your_finnhub_api_key
   ```

3. Save the file (Ctrl+O, then Enter, then Ctrl+X)

## Step 7: Verify the Installation

1. Check if the service is running:
   ```bash
   sudo systemctl status trading-agent
   ```

2. Check the logs:
   ```bash
   tail -f /opt/trading-agent/logs/app.log
   ```

## Step 8: Set Up Monitoring and Alerts

1. In the Google Cloud Console, go to "Monitoring"
2. Create a new dashboard for your trading agent
3. Set up alerts for high CPU usage, memory usage, and service down

## Step 9: Set Up Backup

1. Make the backup script executable:
   ```bash
   chmod +x ~/trading-agent/backup.sh
   ```

2. Copy the script to the appropriate location:
   ```bash
   sudo cp ~/trading-agent/backup.sh /opt/trading-agent/
   ```

3. Set up a cron job:
   ```bash
   sudo crontab -e
   ```

4. Add the following line to run the backup daily at 2 AM:
   ```
   0 2 * * * /opt/trading-agent/backup.sh >> /opt/trading-agent/logs/backup.log 2>&1
   ```

## Cost Optimization

To keep your Google Cloud costs as low as possible:

1. **Use the free tier**:
   - The e2-micro instance is eligible for the free tier
   - You get 1 vCPU and 1 GB memory
   - Free for the first 12 months

2. **Set up budget alerts**:
   - In Google Cloud Console, go to "Billing" > "Budgets & Alerts"
   - Create a budget with alerts at 50%, 90%, and 100% of your expected monthly cost

3. **Optimize resource usage**:
   - The agent only needs to run during market hours
   - You can set up a cron job to start/stop the VM:
     ```bash
     # Start VM at 8:30 AM on weekdays
     30 8 * * 1-5 gcloud compute instances start trading-agent --zone=us-east1-b
     
     # Stop VM at 5:30 PM on weekdays
     30 17 * * 1-5 gcloud compute instances stop trading-agent --zone=us-east1-b
     ```

## Maintenance and Updates

To keep your agent running smoothly:

1. **Regular updates**:
   ```bash
   cd /opt/trading-agent
   git pull
   source venv/bin/activate
   pip install -r requirements.txt
   sudo systemctl restart trading-agent
   ```

2. **Monitor logs**:
   ```bash
   tail -f /opt/trading-agent/logs/app.log
   ``` 