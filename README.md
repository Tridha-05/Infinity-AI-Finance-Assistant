# AI Finance Assistant

A Streamlit-based AI-powered financial assistant that allows users to upload their transaction data, get financial insights, visualize spending trends, and receive encrypted summaries with sentiment analysis. Perfect for users who want a private, smart, and interactive way to manage their personal finances.

## Features

- **Transaction Analysis**: Upload CSV files containing financial transactions and get meaningful summaries.
- **Data Visualization**: Dynamic bar charts of expenses over time for easy trend identification.
- **AI Insights**: Summarized explanations of spending behavior using OpenAI's GPT model.
- **Sentiment Analysis**: Understand the emotional tone of your financial activity.
- **Encryption/Decryption**: Secure your financial summaries with symmetric key encryption.
- **Mock Bank Transactions**: Perform simulated debit/credit transactions and update your balance in real-time.

---

## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/Infinity-AI-Finance-Assistant.git
   cd ai-finance-assistant
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
3. **Install dependencies**
    ```bash
   pip install -r requirements.txt
4. **Run the app**
    ```bash
   streamlit run app.py
    
## Usage

Upload a CSV file with your transaction data (must include columns like date, amount, category, and description).
Use the sidebar to:
Get summarized financial insights.
Encrypt and decrypt your transaction summaries using a secret key.
Simulate deposits and withdrawals.

## Example CSV Format
```
date,amount,category,description
2024-01-01,150,Food,Restaurant dinner
2024-01-02,200,Transport,Monthly metro pass

