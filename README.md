# Trading Agent

An automated trading agent that analyzes market conditions, generates trading suggestions, and posts them to Twitter.

## Features

- Real-time market data analysis
- Technical indicator calculations
- Trading strategy generation
- Automated Twitter posting
- Health monitoring and logging
- Rate limiting for API calls

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:
   ```
   X_API_KEY=your_twitter_api_key
   X_API_SECRET=your_twitter_api_secret
   X_ACCESS_TOKEN=your_twitter_access_token
   X_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   GROK_API_KEY=your_grok_api_key
   NEWSAPI_KEY=your_newsapi_key
   FINNHUB_API_KEY=your_finnhub_api_key
   ```
4. Run the agent: `python main.py`

## Deployment

For deployment instructions, see:
- [Google Cloud Platform Deployment](docs/gcp_deployment.md)
- [Oracle Cloud Deployment](docs/oracle_deployment.md)

## License

MIT 