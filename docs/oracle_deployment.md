# Deploying the Trading Agent on Oracle Cloud

This guide walks you through deploying the trading agent on Oracle Cloud Free Tier, which provides free forever resources.

## Prerequisites

- An Oracle Cloud account (free tier)
- Your trading agent code in a GitHub repository

## Step 1: Create an Oracle Cloud Account

1. Go to [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)
2. Sign up for a free account
3. Verify your email and identity

## Step 2: Create a VM Instance

1. Log into Oracle Cloud Console
2. Go to Compute > Instances
3. Click "Create Instance"
4. Configure the instance:
   - Name: `trading-agent`
   - Image: Canonical Ubuntu 20.04
   - Shape: VM.Standard.A1.Flex (Always Free)
   - Networking: Create a new VCN
   - Public IP: Yes
   - SSH Keys: Generate a new key pair
5. Click "Create"

## Step 3: Set Up SSH Access

1. Download the private key from Oracle Cloud Console
2. Save it to your local machine (e.g., as `oracle_key.pem`)
3. Set the correct permissions:
   ```bash
   chmod 400 oracle_key.pem
   ```
4. Connect to your instance:
   ```bash
   ssh -i oracle_key.pem ubuntu@<your-instance-ip>
   ```

## Step 4: Deploy Your Application

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
   chmod +x ~/trading-agent/deploy.sh
   ```

4. Run the deployment script:
   ```bash
   cd ~/trading-agent
   ./deploy.sh
   ```

## Step 5: Set Up Environment Variables

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

## Step 6: Verify the Installation

1. Check if the service is running:
   ```bash
   sudo systemctl status trading-agent
   ```

2. Check the logs:
   ```bash
   tail -f /opt/trading-agent/logs/app.log
   ```

## Step 7: Set Up Backup

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

Oracle Cloud Free Tier provides:
- 2 AMD-based Compute instances
- 1GB RAM, 50GB storage
- Free forever (no expiration)

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