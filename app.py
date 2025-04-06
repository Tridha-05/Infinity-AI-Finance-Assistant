import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import json
import base64
import hashlib
import os
import time

# Set page configuration
st.set_page_config(page_title="Infinity AI Finance Assistant", layout="wide")

# Initialize sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

# Simple encryption/decryption functions using basic hashing and XOR
def simple_encrypt(data, password):
    """Simple encryption using password hash and XOR"""
    # Convert data to JSON string with date handling
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pd.Timestamp) or isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    json_data = json.dumps(data, cls=DateTimeEncoder)
    data_bytes = json_data.encode('utf-8')
    
    # Create a repeating key from password hash
    hash_obj = hashlib.sha256(password.encode())
    key_bytes = hash_obj.digest()
    
    # Extend key to match data length by repeating it
    extended_key = (key_bytes * (len(data_bytes) // len(key_bytes) + 1))[:len(data_bytes)]
    
    # XOR operation between data and key
    encrypted_bytes = bytes(a ^ b for a, b in zip(data_bytes, extended_key))
    
    # Base64 encode for storage/transmission
    return base64.b64encode(encrypted_bytes).decode('utf-8')

def simple_decrypt(encrypted_data, password):
    """Simple decryption using password hash and XOR"""
    try:
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        # Create key from password hash
        hash_obj = hashlib.sha256(password.encode())
        key_bytes = hash_obj.digest()
        
        # Extend key to match data length
        extended_key = (key_bytes * (len(encrypted_bytes) // len(key_bytes) + 1))[:len(encrypted_bytes)]
        
        # XOR operation to decrypt
        decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, extended_key))
        
        # Parse JSON
        json_data = decrypted_bytes.decode('utf-8')
        decrypted_data = json.loads(json_data)
        
        # Convert ISO format dates back to datetime if needed
        for item in decrypted_data:
            if 'Date' in item and isinstance(item['Date'], str):
                try:
                    item['Date'] = pd.to_datetime(item['Date'])
                except:
                    pass  # Keep as string if conversion fails
                    
        return decrypted_data
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        return None

# Mock Bank API
class MockBankAPI:
    def __init__(self):
        self.accounts = {
            "12345": {"name": "Checking Account", "balance": 2500.00},
            "67890": {"name": "Savings Account", "balance": 8500.00}
        }
        self.transactions = self._generate_mock_transactions()
    
    def _generate_mock_transactions(self):
        """Generate realistic mock transactions"""
        transactions = []
        
        # Transaction templates
        expense_templates = [
            {"description": "Grocery Store", "min": 20, "max": 200, "category": "Food"},
            {"description": "Restaurant Meal", "min": 15, "max": 120, "category": "Food"},
            {"description": "Coffee Shop", "min": 3, "max": 15, "category": "Food"},
            {"description": "Gas Station", "min": 20, "max": 60, "category": "Transportation"},
            {"description": "Uber Ride", "min": 10, "max": 40, "category": "Transportation"},
            {"description": "Amazon Purchase", "min": 10, "max": 150, "category": "Shopping"},
            {"description": "Target Purchase", "min": 15, "max": 100, "category": "Shopping"},
            {"description": "Electricity Bill", "min": 50, "max": 150, "category": "Utilities"},
            {"description": "Internet Bill", "min": 40, "max": 100, "category": "Utilities"},
            {"description": "Phone Bill", "min": 50, "max": 120, "category": "Utilities"},
            {"description": "Rent Payment", "min": 800, "max": 2000, "category": "Housing"},
            {"description": "Doctor Visit", "min": 20, "max": 200, "category": "Healthcare"},
            {"description": "Pharmacy", "min": 10, "max": 80, "category": "Healthcare"},
            {"description": "Movie Tickets", "min": 15, "max": 50, "category": "Entertainment"},
            {"description": "Netflix Subscription", "min": 10, "max": 20, "category": "Entertainment"},
            {"description": "Gym Membership", "min": 30, "max": 80, "category": "Healthcare"}
        ]
        
        income_templates = [
            {"description": "Salary Deposit", "min": 2000, "max": 5000, "category": "Income"},
            {"description": "Freelance Payment", "min": 200, "max": 1000, "category": "Income"},
            {"description": "Interest Income", "min": 5, "max": 100, "category": "Income"}
        ]
        
        # Generate transactions for the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        current_date = start_date
        
        # Add paycheck every 2 weeks
        payday = start_date + timedelta(days=(4 - start_date.weekday()) % 7)  # First Friday
        while payday <= end_date:
            amount = random.uniform(2800, 3200)
            transactions.append({
                "date": payday.strftime("%Y-%m-%d"),
                "description": "Salary Deposit",
                "amount": amount,
                "category": "Income"
            })
            payday += timedelta(days=14)  # Biweekly
        
        # Add rent on the 1st of each month
        rent_day = datetime(start_date.year, start_date.month, 1)
        if rent_day < start_date:
            rent_day = datetime(start_date.year, start_date.month + 1, 1)
        
        while rent_day <= end_date:
            transactions.append({
                "date": rent_day.strftime("%Y-%m-%d"),
                "description": "Rent Payment",
                "amount": -1500.00,
                "category": "Housing"
            })
            month = rent_day.month + 1
            year = rent_day.year
            if month > 12:
                month = 1
                year += 1
            rent_day = datetime(year, month, 1)
        
        # Add random expenses and occasional income
        while current_date <= end_date:
            # Random number of transactions per day (0-5)
            num_transactions = random.randint(0, 3)
            
            for _ in range(num_transactions):
                # 90% chance of expense, 10% chance of income
                if random.random() < 0.9:
                    template = random.choice(expense_templates)
                    amount = -random.uniform(template["min"], template["max"])
                else:
                    template = random.choice(income_templates)
                    amount = random.uniform(template["min"], template["max"])
                
                transactions.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "description": template["description"],
                    "amount": round(amount, 2),
                    "category": template["category"]
                })
            
            current_date += timedelta(days=1)
            
        return sorted(transactions, key=lambda x: x["date"])
    
    def get_accounts(self, user_id="demo"):
        """Get account information for a user"""
        return self.accounts
    
    def get_balance(self, account_id):
        """Get balance for a specific account"""
        if account_id in self.accounts:
            return self.accounts[account_id]["balance"]
        return None
    
    def get_transactions(self, start_date=None, end_date=None, account_id=None):
        """Get transactions for a date range"""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        filtered_transactions = [
            t for t in self.transactions 
            if start_date <= t["date"] <= end_date
        ]
        
        return filtered_transactions
    
    def to_dataframe(self, transactions):
        """Convert transactions to pandas DataFrame"""
        df = pd.DataFrame(transactions)
        return df

# Function to categorize transactions based on keywords
def categorize_transaction(description):
    description = description.lower()
    
    categories = {
        'Food': ['restaurant', 'cafe', 'grocery', 'food', 'meal', 'dinner', 'lunch', 'breakfast'],
        'Transportation': ['gas', 'uber', 'lyft', 'taxi', 'bus', 'train', 'transit', 'fuel', 'car'],
        'Entertainment': ['movie', 'theatre', 'concert', 'ticket', 'netflix', 'spotify', 'subscription'],
        'Shopping': ['amazon', 'walmart', 'target', 'store', 'buy', 'purchase', 'shop'],
        'Utilities': ['electricity', 'water', 'internet', 'phone', 'bill', 'utility'],
        'Housing': ['rent', 'mortgage', 'housing', 'apartment', 'maintenance'],
        'Healthcare': ['doctor', 'medical', 'pharmacy', 'health', 'dental', 'medicine'],
        'Income': ['salary', 'deposit', 'paycheck', 'payment received', 'income', 'refund'],
    }
    
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    
    return 'Other'

# Function to analyze transactions
def analyze_transactions(df):
    # Add categories if not present
    if 'Category' not in df.columns:
        df['Category'] = df['Description'].apply(categorize_transaction)
    
    # Convert date strings to datetime if needed
    if df['Date'].dtype == 'object':
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            pass  # Keep as string if conversion fails
    
    # Identify expense transactions (negative amounts)
    expenses = df[df['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()  # Make positive for easier analysis
    
    # Income transactions
    income = df[df['Amount'] > 0]
    
    return df, expenses, income

# Function to generate AI insights
def generate_ai_insights(df, expenses, income):
    # Prepare basic statistics
    total_income = income['Amount'].sum()
    total_expenses = expenses['Amount'].sum()
    net_flow = total_income - total_expenses
    
    if len(expenses) > 0:
        top_expenses = expenses.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(3)
        top_expense_categories = top_expenses.index.tolist()
    else:
        top_expense_categories = []
    
    # Generate description for sentiment analysis
    description = f"""
    Financial summary: Total income ${total_income:.2f}, total expenses ${total_expenses:.2f}, 
    net cash flow ${net_flow:.2f}. Top spending categories: {', '.join(top_expense_categories)}.
    """
    
    # Run sentiment analysis
    sentiment_result = sentiment_model(description)
    financial_health = sentiment_result[0]['label']
    confidence = sentiment_result[0]['score']
    
    # Generate simple advice based on financial situation
    advice = []
    
    if net_flow < 0:
        advice.append("❌ Your expenses exceed your income. Consider cutting back on spending.")
        
        if len(top_expense_categories) > 0:
            advice.append(f"📊 Focus on reducing spending in your top category: {top_expense_categories[0]}.")
    else:
        advice.append("✅ Your income exceeds your expenses, which is positive.")
        advice.append(f"💰 Consider saving or investing your surplus of ${net_flow:.2f}.")
    
    if total_expenses > 0:
        # Find the category with the highest growth
        try:
            # Try to group by month and category
            if isinstance(df['Date'].iloc[0], datetime):
                expenses['Month'] = expenses['Date'].dt.strftime('%Y-%m')
                monthly_category = expenses.groupby(['Month', 'Category'])['Amount'].sum().reset_index()
                
                if len(monthly_category['Month'].unique()) > 1:
                    # Get last two months if available
                    last_months = sorted(monthly_category['Month'].unique())[-2:]
                    
                    if len(last_months) > 1:
                        last_month = monthly_category[monthly_category['Month'] == last_months[1]]
                        prev_month = monthly_category[monthly_category['Month'] == last_months[0]]
                        
                        # Find categories that increased
                        merged = pd.merge(last_month, prev_month, on='Category', suffixes=('_last', '_prev'))
                        merged['growth'] = merged['Amount_last'] - merged['Amount_prev']
                        
                        if not merged.empty:
                            top_growth = merged.sort_values('growth', ascending=False).iloc[0]
                            if top_growth['growth'] > 0:
                                advice.append(f"⚠️ Your spending in {top_growth['Category']} increased by ${top_growth['growth']:.2f}.")
        except:
            pass
    
    # Add savings advice
    if 'Housing' in top_expense_categories:
        advice.append("🏠 Housing is a major expense. Consider if there are ways to reduce these costs.")
    
    if 'Food' in top_expense_categories:
        advice.append("🍲 Consider meal planning to reduce food costs.")
    
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'net_flow': net_flow,
        'financial_health': financial_health,
        'confidence': confidence,
        'advice': advice
    }

# Initialize session state
if 'bank_api' not in st.session_state:
    st.session_state.bank_api = MockBankAPI()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# App title and description
st.title("💰 AI Finance Assistant")
st.markdown("Connect to your bank account or upload transaction data to get AI-powered insights.")

# Authentication
if not st.session_state.authenticated:
    st.header("🔐 Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Connect to Bank")
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", value="demo", type="password")
        
        if st.button("Login"):
            # In a real app, you would validate credentials
            # Here we just accept any input for demo
            st.session_state.authenticated = True
            st.success("Successfully logged in!")
            st.rerun()
    
    with col2:
        st.subheader("Manual Data")
        # File uploader option still available
        uploaded_file = st.file_uploader("Upload your transaction CSV file", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ["Date", "Description", "Amount"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"CSV file is missing required columns: {', '.join(missing_cols)}")
                else:
                    st.session_state.df = df
                    st.session_state.authenticated = True
                    st.success("File uploaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading the file: {e}")
                
else:
    # User is authenticated, show main content
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔒 Secure Data", "⚙️ Settings"])
    
    with tab1:
        # Sidebar for bank connection
        st.sidebar.header("📱 Bank Connection")
        
        # Account selection
        accounts = st.session_state.bank_api.get_accounts()
        account_options = [f"{acc_id}: {acc_info['name']}" for acc_id, acc_info in accounts.items()]
        
        selected_account = st.sidebar.selectbox("Select Account", account_options)
        account_id = selected_account.split(":")[0].strip()
        
        # Date range selection
        st.sidebar.subheader("Date Range")
        date_options = ["Last 30 days", "Last 60 days", "Last 90 days", "Custom"]
        date_selection = st.sidebar.radio("Select period", date_options)
        
        end_date = datetime.now()
        
        if date_selection == "Last 30 days":
            start_date = end_date - timedelta(days=30)
        elif date_selection == "Last 60 days":
            start_date = end_date - timedelta(days=60)
        elif date_selection == "Last 90 days":
            start_date = end_date - timedelta(days=90)
        else:  # Custom
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start date", end_date - timedelta(days=30))
            with col2:
                end_date = st.date_input("End date", end_date)
        
        start_date_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date.strftime("%Y-%m-%d")
        
        # Button to fetch data
        if st.sidebar.button("Fetch Transactions"):
            with st.spinner("Fetching data from bank..."):
                # Simulate API delay
                time.sleep(1)
                transactions = st.session_state.bank_api.get_transactions(
                    start_date=start_date_str, 
                    end_date=end_date_str
                )
                
                # Convert to DataFrame
                df = st.session_state.bank_api.to_dataframe(transactions)
                df.rename(columns={"date": "Date", "description": "Description", "amount": "Amount", "category": "Category"}, inplace=True)
                
                # Store in session state
                st.session_state.df = df
                st.success(f"Fetched {len(df)} transactions")
        
        # Use sample data option
        use_sample = st.sidebar.checkbox("Use sample data instead")
        if use_sample:
            # Create sample data
            df = pd.DataFrame({
                'Date': pd.date_range(start='2025-01-01', periods=20).astype(str),
                'Description': [
                    'Salary Deposit', 'Rent Payment', 'Grocery Store', 'Restaurant Dinner', 
                    'Gas Station', 'Amazon Purchase', 'Phone Bill', 'Movie Tickets',
                    'Electricity Bill', 'Target Shopping', 'Pharmacy', 'Uber Ride',
                    'Coffee Shop', 'Gym Membership', 'Doctor Visit', 'Netflix Subscription',
                    'Car Insurance', 'Grocery Store', 'Restaurant Lunch', 'Interest Income'
                ],
                'Amount': [
                    3000.00, -1200.00, -150.00, -75.00, 
                    -45.00, -120.00, -80.00, -30.00,
                    -90.00, -65.00, -25.00, -18.00,
                    -5.00, -50.00, -100.00, -15.00,
                    -120.00, -130.00, -35.00, 50.00
                ],
                'Category': [
                    'Income', 'Housing', 'Food', 'Food',
                    'Transportation', 'Shopping', 'Utilities', 'Entertainment',
                    'Utilities', 'Shopping', 'Healthcare', 'Transportation',
                    'Food', 'Healthcare', 'Healthcare', 'Entertainment',
                    'Transportation', 'Food', 'Food', 'Income'
                ]
            })
            st.session_state.df = df
            st.success("Sample data loaded!")
            
        # Process and display data if available
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Display raw data
            with st.expander("View Transaction Data"):
                st.dataframe(df)
            
            # Process the data
            df, expenses, income = analyze_transactions(df)
            
            # Generate AI insights
            insights = generate_ai_insights(df, expenses, income)
            
            # Display insights
            st.header("💡 AI Financial Insights")
            
            # Create columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Income", f"${insights['total_income']:.2f}", delta=None)
                
            with col2:
                st.metric("Expenses", f"${insights['total_expenses']:.2f}", delta=None)
                
            with col3:
                delta_color = "normal" if insights['net_flow'] >= 0 else "inverse"
                st.metric("Net Flow", f"${insights['net_flow']:.2f}", 
                        delta=f"{'Positive' if insights['net_flow'] >= 0 else 'Negative'} cash flow", 
                        delta_color=delta_color)
            
            # Display AI sentiment analysis
            st.subheader("Financial Health Assessment")
            st.info(f"AI assessment: {insights['financial_health']} (confidence: {insights['confidence']:.2f})")
            
            # Display advice
            st.subheader("AI Recommendations")
            for advice in insights['advice']:
                st.markdown(f"- {advice}")
            
            # Visualizations
            st.header("📊 Spending Analysis")
            
            # Create tabs for different charts
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Spending by Category", "Income vs. Expenses", "Transaction Timeline"])
            
            with chart_tab1:
                # Spending by category pie chart
                if not expenses.empty:
                    fig = px.pie(
                        expenses.groupby('Category')['Amount'].sum().reset_index(),
                        values='Amount',
                        names='Category',
                        title='Spending by Category',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No expense data available to visualize.")
            
            with chart_tab2:
                # Income vs Expenses bar chart
                income_by_category = income.groupby('Category')['Amount'].sum().reset_index()
                expense_by_category = expenses.groupby('Category')['Amount'].sum().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=income_by_category['Category'],
                    y=income_by_category['Amount'],
                    name='Income',
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    x=expense_by_category['Category'],
                    y=expense_by_category['Amount'],
                    name='Expenses',
                    marker_color='crimson'
                ))
                
                fig.update_layout(
                    title='Income vs Expenses by Category',
                    barmode='group',
                    xaxis_title='Category',
                    yaxis_title='Amount ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab3:
                # Transaction timeline
                try:
                    if isinstance(df['Date'].iloc[0], datetime) or pd.to_datetime(df['Date'], errors='coerce').notna().all():
                        if not isinstance(df['Date'].iloc[0], datetime):
                            df['Date'] = pd.to_datetime(df['Date'])
                        
                        # Sort by date
                        df_sorted = df.sort_values('Date')
                        
                        # Create a time series plot
                        fig = px.line(
                            df_sorted,
                            x='Date',
                            y='Amount',
                            color='Category',
                            title='Transaction Timeline',
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Date format not recognized for timeline visualization.")
                except:
                    st.write("Could not create timeline visualization with the provided date format.")
            
            # Budget recommendations
            st.header("💳 Simple Budget Recommendation")
            
            # Create simple 50/30/20 budget recommendation
            if insights['total_income'] > 0:
                monthly_income = insights['total_income']
                
                # 50/30/20 rule
                needs = monthly_income * 0.5
                wants = monthly_income * 0.3
                savings = monthly_income * 0.2
                
                budget_data = pd.DataFrame({
                    'Category': ['Needs (50%)', 'Wants (30%)', 'Savings (20%)'],
                    'Amount': [needs, wants, savings]
                })
                
                fig = px.bar(
                    budget_data,
                    x='Category',
                    y='Amount',
                    title='Recommended Budget Allocation (50/30/20 Rule)',
                    color='Category',
                    color_discrete_map={
                        'Needs (50%)': 'royalblue',
                        'Wants (30%)': 'lightblue',
                        'Savings (20%)': 'green'
                    }
                )
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display budget breakdown
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Needs (50%)")
                    st.write("Examples:")
                    st.write("- Housing/Rent")
                    st.write("- Groceries")
                    st.write("- Utilities")
                    st.write("- Insurance")
                    st.metric("Budget", f"${needs:.2f}")
                    
                with col2:
                    st.subheader("Wants (30%)")
                    st.write("Examples:")
                    st.write("- Dining out")
                    st.write("- Entertainment")
                    st.write("- Hobbies")
                    st.write("- Shopping")
                    st.metric("Budget", f"${wants:.2f}")
                    
                with col3:
                    st.subheader("Savings (20%)")
                    st.write("Examples:")
                    st.write("- Emergency fund")
                    st.write("- Retirement")
                    st.write("- Debt repayment")
                    st.write("- Investments")
                    st.metric("Budget", f"${savings:.2f}")
        else:
            st.info("Please fetch data from your bank account or use the sample data to see AI insights.")
    
    with tab2:
        st.header("🔒 Secure Your Financial Data")
        
        if 'df' in st.session_state:
            data_to_encrypt = st.session_state.df.to_dict('records')
            
            # Password for encryption
            encryption_password = st.text_input("Set encryption password", type="password", 
                                              help="Set a strong password to encrypt your data")
            
            if st.button("Encrypt and Save Data") and encryption_password:
                # Encrypt data
                encrypted_data = simple_encrypt(data_to_encrypt, encryption_password)
                
                # Create download button
                st.download_button(
                    label="Download Encrypted Data",
                    data=encrypted_data,
                    file_name="encrypted_finance_data.txt",
                    mime="text/plain"
                )
                
                st.success("Data encrypted successfully! You can decrypt it later using the same password.")
            
            # Section to decrypt data
            st.subheader("Decrypt Data")
            
            uploaded_encrypted = st.file_uploader("Upload encrypted data file", type=["txt"])
            decrypt_password = st.text_input("Enter decryption password", type="password")
            
            if uploaded_encrypted and decrypt_password and st.button("Decrypt Data"):
                try:
                    # Read encrypted data
                    encrypted_data = uploaded_encrypted.getvalue().decode('utf-8')
                    
                    # Decrypt data
                    decrypted_data = simple_decrypt(encrypted_data, decrypt_password)
                    
                    if decrypted_data:
                        # Convert to DataFrame
                        decrypted_df = pd.DataFrame(decrypted_data)
                        
                        # Store in session state
                        st.session_state.df = decrypted_df
                        
                        st.success("Data decrypted successfully!")
                        st.dataframe(decrypted_df)
                except Exception as e:
                    st.error(f"Error decrypting data: {e}")
        else:
            st.info("Please load transaction data first to use the encryption features.")
    
    with tab3:
        st.header("⚙️ Settings")
        
        st.subheader("Bank Connection")
        st.warning("For demonstration purposes only. In a real app, this would connect to actual bank APIs.")
        
        # Mock connection status
        st.success("Connected to Demo Bank")
        
        if st.button("Disconnect"):
            st.session_state.authenticated = False
            st.rerun()
            
        st.subheader("Data Privacy")
        st.info("""
        This application uses simple encryption to protect your financial data. 
        When you encrypt your data:
        - Only you can access it with your password
        - Data is never stored on our servers
        - Your financial information remains private
        """)
        
        # Clear data option
        if st.button("Clear All Data"):
            if 'df' in st.session_state:
                del st.session_state.df
            st.success("All data has been cleared from the application.")
            time.sleep(1)
            st.rerun()