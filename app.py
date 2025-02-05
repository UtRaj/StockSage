# Importing Libraries
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import streamlit as st
from crewai.tools import tool
from datetime import datetime
from textblob import TextBlob
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from crewai import Crew, Agent, Task, Process, LLM
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables from the .env file
load_dotenv()

# Set the Streamlit Page Configuration
st.set_page_config(page_title="StockSage", layout="wide")

# Access the API key from the environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Sidebar for model selection
st.sidebar.header("Select AI Model")
selected_model = st.sidebar.radio("Choose a model:", ("Gemini", "Groq"))

# Initialize LLM based on user selection
if selected_model == "Gemini":
    llm_llama70b = LLM(model="gemini/gemini-pro", temperature=0.7)
    st.success("Google Gemini Pro has been Successfully accessed!!")
elif selected_model == "Groq":
    llm_llama70b = LLM(model="groq/gemma2-9b-it", temperature=0.7)  
    st.success("Google Gemma2-9b-it from Groq has been Successfully accessed!!")

plot_generated = False


@retry(stop=stop_after_attempt(3), wait=wait_fixed(12))
# Define the function with retry logic and only one plot generation
@tool("get_basic_stock_info")
def get_basic_stock_info(ticker: str, retries=3, delay=5) -> pd.DataFrame:
    """Retrieve basic stock information for a given ticker."""
    global plot_generated  # Use global flag to track if plot is generated
    
    attempt = 0
    while attempt < retries:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            basic_info = pd.DataFrame({
                'Name': [info.get('longName', 'N/A')],
                'Sector': [info.get('sector', 'N/A')],
                'Industry': [info.get('industry', 'N/A')],
                'Market Cap': [info.get('marketCap', 'N/A')],
                'Current Price': [info.get('currentPrice', 'N/A')],
                '52 Week High': [info.get('fiftyTwoWeekHigh', 'N/A')],
                '52 Week Low': [info.get('fiftyTwoWeekLow', 'N/A')],
                'Average Volume': [info.get('averageVolume', 'N/A')]
            })

            # Only generate the plot if it hasn't been done before
            if not plot_generated:
                history_5 = stock.history(period="5d")['Close']
                history_30 = stock.history(period="1mo")['Close']



                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                sns.lineplot(x=history_5.index, y=history_5.values, ax=axes[0], label="Stock Price (5 days)")
                axes[0].plot(history_5.index, history_5.values, label="Stock Price (5 days)")
                axes[0].set_title('Stock Price (Last 5 Days)')
                axes[0].set_xlabel('Date')
                axes[0].set_ylabel('Price ($)')

                sns.lineplot(x=history_30.index, y=history_30.values, ax=axes[1], label="Stock Price (30 days)", color='orange')
                axes[1].plot(history_30.index, history_30.values, label="Stock Price (30 days)", color='orange')
                axes[1].set_title('Stock Price (Last 30 Days)')
                axes[1].set_xlabel('Date')
                axes[1].set_ylabel('Price ($)')
                
                plt.tight_layout()
                st.pyplot(fig)
                plot_generated = True  # Set flag to True after the plot is generated

            return basic_info, history_5, history_30
        except Exception as e:
            print(f"Error retrieving stock info or plotting chart: {e}")
            attempt += 1
            time.sleep(delay)
    
    return pd.DataFrame()  # Return empty DataFrame if all retries fail


@retry(stop=stop_after_attempt(3), wait=wait_fixed(12))
# Initialize Tools for Fundamental Analysis for given stocks for specific period
@tool("get_fundamental_analysis")
def get_fundamental_analysis(ticker: str, period: str = '1y') -> pd.DataFrame:
    """Retrieve Fundamental Analysis for given stocks for specific period."""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        info = stock.info
        
        fundamental_analysis = pd.DataFrame({
            'PE Ratio': [info.get('trailingPE', 'N/A')],
            'Forward PE': [info.get('forwardPE', 'N/A')],
            'PEG Ratio': [info.get('pegRatio', 'N/A')],
            'Price to Book': [info.get('priceToBook', 'N/A')],
            'Dividend Yield': [info.get('dividendYield', 'N/A')],
            'EPS (TTM)': [info.get('trailingEps', 'N/A')],
            'Revenue Growth': [info.get('revenueGrowth', 'N/A')],
            'Profit Margin': [info.get('profitMargins', 'N/A')],
            'Free Cash Flow': [info.get('freeCashflow', 'N/A')],
            'Debt to Equity': [info.get('debtToEquity', 'N/A')],
            'Return on Equity': [info.get('returnOnEquity', 'N/A')],
            'Operating Margin': [info.get('operatingMargins', 'N/A')],
            'Quick Ratio': [info.get('quickRatio', 'N/A')],
            'Current Ratio': [info.get('currentRatio', 'N/A')],
            'Earnings Growth': [info.get('earningsGrowth', 'N/A')],
            'Stock Price Avg (Period)': [history['Close'].mean()],
            'Stock Price Max (Period)': [history['Close'].max()],
            'Stock Price Min (Period)': [history['Close'].min()]
        })
        
        return fundamental_analysis
    except Exception as e:
        st.error(f"Error retrieving fundamental analysis: {e}")
        return pd.DataFrame()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(12))
# Initialize Tools for Stock Risk Assessment
@tool("get_stock_risk_assessment")
def get_stock_risk_assessment(ticker: str, period: str = "3mo", market_ticker: str = "^GSPC") -> pd.DataFrame:
    """Retrieve comprehensive Stock Risk Assessment including Beta, Volatility, VaR, Drawdown, Sharpe & Sortino Ratios."""
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)

        # Ensure sufficient data
        if history.empty or len(history) < 10:
            st.warning(f"Insufficient data for {ticker}.")
            return pd.DataFrame()

        # Calculate daily returns
        returns = history['Close'].pct_change().dropna()
        if returns.empty:
            st.warning(f"Not enough return data to analyze {ticker}.")
            return pd.DataFrame()

        # Fetch market data for Beta calculation
        market = yf.Ticker(market_ticker)
        market_history = market.history(period=period)
        market_returns = market_history['Close'].pct_change().dropna()
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        beta = (aligned_data.cov().iloc[0, 1] / market_returns.var()) if not aligned_data.empty else 'N/A'


        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)
        var_95 = np.percentile(returns, 5)
        max_drawdown = ((history['Close'] - history['Close'].cummax()) / history['Close'].cummax()).min()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 'N/A'
        
        # Calculate Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        sortino_ratio = np.sqrt(252) * (returns.mean() / downside_deviation) if downside_deviation != 0 else 'N/A'

        # Plot returns distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(returns, kde=True, ax=ax)
        ax.set_title('Stock Returns Distribution')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Return results as DataFrame
        return pd.DataFrame({
            'Annualized Volatility': [volatility],
            'Beta': [beta],
            'Value at Risk (95%)': [var_95],
            'Maximum Drawdown': [max_drawdown],
            'Sharpe Ratio': [sharpe_ratio],
            'Sortino Ratio': [sortino_ratio]
        })

    except Exception as e:
        st.error(f"Error retrieving risk assessment for {ticker}: {e}")
        return pd.DataFrame()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(12))
# Initialize Tool for performing Technical Analysis
@tool("get_technical_analysis")
def get_technical_analysis(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Retrieve Technical Analysis for given stocks."""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)

        if history.empty or len(history) < 50:
            st.warning(f"Insufficient data for {ticker}. At least 50 data points are required.")
            return pd.DataFrame()

        # Calculate indicators
        history['SMA_50'] = history['Close'].rolling(window=50).mean()
        history['SMA_200'] = history['Close'].rolling(window=200).mean()
        history['MACD'] = history['Close'].ewm(span=12, adjust=False).mean() - history['Close'].ewm(span=26, adjust=False).mean()
        history['Signal'] = history['MACD'].ewm(span=9, adjust=False).mean()
        delta = history['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        history['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # Get latest values
        latest = history.iloc[-1]
        trend = "Bullish" if latest['Close'] > latest['SMA_50'] > latest['SMA_200'] else "Bearish" if latest['Close'] < latest['SMA_50'] < latest['SMA_200'] else "Neutral"
        macd_signal = "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish"
        rsi_signal = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"

        return pd.DataFrame({
            'Indicator': ['Current Price', '50-day SMA', '200-day SMA', 'RSI (14-day)', 'MACD', 'MACD Signal', 'Trend', 'MACD Signal', 'RSI Signal'],
            'Value': [
                f'${latest["Close"]:.2f}', f'${latest["SMA_50"]:.2f}', f'${latest["SMA_200"]:.2f}', f'{latest["RSI"]:.2f}',
                f'{latest["MACD"]:.2f}', f'{latest["Signal"]:.2f}', trend, macd_signal, rsi_signal
            ]
        })
    except Exception as e:
        st.error(f"Error retrieving technical analysis: {e}")
        return pd.DataFrame()



@retry(stop=stop_after_attempt(3), wait=wait_fixed(12))
# Initialize Stock news Tool
@tool("get_stock_news")
def get_stock_news(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Retrieve recent news articles related to the stock."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:limit]
        sentiments = []
        for article in news:
            text = article['title']
            sentiment = TextBlob(text).sentiment.polarity
            sentiment_label = 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
            sentiments.append(sentiment_label)
        
        news_data = []
        for article, sentiment in zip(news, sentiments):
            news_entry = {
                "Title": article['title'],
                "Publisher": article['publisher'],
                "Published": datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S'),
                "Link": article['link'],
                "Sentiment": sentiment
            }
            news_data.append(news_entry)
        
        return pd.DataFrame(news_data)
    except Exception as e:
        st.error(f"Error retrieving stock news: {e}")
        return pd.DataFrame()



## Initialize all the Agents     
stock_researcher = Agent(
    llm=llm_llama70b,
    role="Stock Researcher",
    goal="Identify the stock and the stock ticker, and if you already have the stock ticker and if it's necessary, get basic stock info about the selected stock.",
    backstory="A junior stock researcher with a knack for gathering relevant, basic information about stocks, the relevant company/companies, the industry, and some basic info about the stock's performance.",
    tools=[get_basic_stock_info, get_fundamental_analysis, get_technical_analysis, get_stock_news],
    verbose=True,
    allow_delegation=False
)


financial_analyst = Agent(
    llm=llm_llama70b,
    role="Financial Analyst",
    goal="Perform in-depth fundamental and technical analysis on the stock, focusing on aspects most relevant to the user's query.",
    backstory="A seasoned financial analyst with expertise in interpreting complex financial data and translating it into insights tailored to various levels of financial literacy.",
    tools=[get_technical_analysis, get_fundamental_analysis, get_stock_risk_assessment],
    verbose=True,
    allow_delegation=False
)



news_analyst = Agent(
    llm=llm_llama70b,
    role="News Analyst",
    goal="Fetch recent news articles related to the stock and analyze their potential impact on performance.",
    backstory="A sharp news analyst who can quickly digest information, assess its relevance to stock performance, and provide concise summaries.",
    tools=[get_stock_news],
    verbose=True,
    allow_delegation=False
)


report_writer = Agent(
    llm=llm_llama70b,
    role='Financial Report Writer',
    goal='Synthesize all analysis into a cohesive, professional stock report.',
    backstory='An experienced financial writer with a talent for clear, concise reporting.',
    tools=[],
    verbose=True,
    allow_delegation=False
)


## Initialize all the Tasks 
collect_stock_info = Task(
    description='''
    1. Extract the ticker of the stock (or stocks) mentioned in the user query as well as the timeframe (if mentioned). If the ticker is not provided, use the query to identify the stock ticker.
    2. If the query implies a novice user, prepare brief explanations for key financial terms. If nothing is mentioned, assume that the user has an above-average understanding of financial terms.
    
    Expect only basic stock info from this task.
    
    User query: {query}.
    
    Your response should be on the basis of:
    Ticker: [identified stock ticker]
    Timeframe: [identified timeframe]
    Analysis Focus: [identified focus of analysis]
    User Expertise: [implied level of financial expertise]
    Key Concerns: [specific concerns or priorities mentioned]
    ''',
    expected_output="A summary of the stock's key financial metrics and performance, tailored to the user's query.",
    agent=stock_researcher,
    dependencies=[],
    context=[]
)


perform_analysis = Task(
    description='''
    Conduct a thorough analysis of the stock, tailored to the user's query and expertise level.
    1. Use the get_stock_info, get_fundamental_analysis, get_stock_risk_assessment, and get_technical_analysis tools as needed, based on the query's focus. E.g., if the query is about the fundamentals of a stock, then technical info need not be present.
    2. Focus on metrics and trends most relevant to the user's specific question and identified timeframe.
    3. Provide clear explanations of complex financial concepts if the query suggests a novice user.
    4. Relate the analysis directly to the key concerns identified in the query interpretation.
    5. Consider both historical performance and future projections in your analysis.
    
    User query: {query}.
    ''',
    expected_output="A detailed analysis of the stock's financial and/or technical performance, directly addressing the user's query and concerns.",
    agent=financial_analyst,
    dependencies=[collect_stock_info],
    context=[collect_stock_info]
)


analyze_stock_news = Task(
    description='''
    1. Use the get_stock_news tool to fetch recent news related to the stock.
    2. Analyze the sentiment and potential impact of the news on the stock's performance.
    3. Conclude with an overall assessment of how recent news might influence the stock in the relevant timeframe.
    
    NOTE: Re-fetching news will get you the same results.
    ''',
    expected_output="A summary of recent news articles related to the stock and their potential impact on performance.",
    agent=news_analyst,
    dependencies=[collect_stock_info],
    context=[collect_stock_info]
)


generate_stock_report = Task(
    description='''
    Synthesize all the collected information and analyses into a stock report tailored to the user's specific query.
    The report should:
    1. Begin with an Executive Summary that directly addresses the user's question
    2. Include relevant sections based on the query's focus
    3. Provide an Investment Recommendation that specifically answers the user's query
    4. Conclude with a summary that ties all insights back to the original question

    Ensure that:
    - The report directly answers the user's specific question
    - The language and depth of analysis match the user's level of expertise implied by the query
    - The report highlights factors most relevant to the user's identified concerns and timeframe
    - Clear, professional language is used throughout, with well-reasoned insights
    - The report is in markdown format for easy reading and formatting
    - The report should be crisp but detailed. You can reiterate important points but avoid redundancy.
    - You are an expert in the field, so you should be confident in your answer, requiring no further action/analysis from the user. It is your job to give a clear recommendation.
    - The report should contain only the relevant info. E.g. if the query is about the fundamentals of a stock, then technical info need not be present.
    
    User query: {query}.
    ''',
    expected_output="A comprehensive stock report in markdown format, addressing all aspects of the user's query and providing a clear investment recommendation.",
    agent=report_writer,
    dependencies=[collect_stock_info],
    context=[collect_stock_info, perform_analysis, analyze_stock_news],
    callback=lambda x: (st.markdown("### Sentiment Analysis of News"), st.write(x))  # Add sentiment analysis
)


## Initialize the Crew Function 
crew = Crew(
    agents=[stock_researcher, financial_analyst, news_analyst, report_writer],
    tasks=[
        collect_stock_info,
        perform_analysis,
        analyze_stock_news,
        generate_stock_report
    ],
    process=Process.sequential,
    manager_llm=llm_llama70b
)


st.title("StockSage: Advanced Stock Analysis Platform")

st.sidebar.header("Enter your stock analysis question")
query = st.sidebar.text_area(label="Your Query:", value="Should Nvidia be a safe bet for long-term investment?", height=100)
analyze_button = st.sidebar.button("Analyze")

if analyze_button:
    st.info(f"Wait for a few minutes to analyze the query: {query}.")

    default_date = datetime.now().date()
    result = crew.kickoff(inputs={"query": query, "default_date": str(default_date)})
    
    st.success("Analysis Completed!")
    
    st.markdown("## Report Generated!!")
    st.markdown(result)






