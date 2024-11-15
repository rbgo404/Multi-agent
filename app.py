import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from dotenv import load_dotenv
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from crewai_tools import LlamaIndexTool
from langchain_ollama import ChatOllama
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai_tools import tool
from typing import Dict, Any
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsInput
from ollama_utils import start_and_check_ollama

@tool
def yf_fundamental_analysis(ticker: str):
    """
        Perform a comprehensive fundamental analysis on the given stock symbol.
    
        Args:
            stock_symbol (str): The stock symbol to analyze.
    
        Returns:
            dict: A dictionary with the detailed fundamental analysis results.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Data processing
        financials = stock.financials.infer_objects(copy=False)
        balance_sheet = stock.balance_sheet.infer_objects(copy=False)
        cash_flow = stock.cashflow.infer_objects(copy=False)

        # Fill missing values
        financials = financials.ffill()
        balance_sheet = balance_sheet.ffill()
        cash_flow = cash_flow.ffill()

        # Key Ratios and Metrics
        ratios = {
            "P/E Ratio": info.get('trailingPE'),
            "Forward P/E": info.get('forwardPE'),
            "P/B Ratio": info.get('priceToBook'),
            "P/S Ratio": info.get('priceToSalesTrailing12Months'),
            "PEG Ratio": info.get('pegRatio'),
            "Debt to Equity": info.get('debtToEquity'),
            "Current Ratio": info.get('currentRatio'),
            "Quick Ratio": info.get('quickRatio'),
            "ROE": info.get('returnOnEquity'),
            "ROA": info.get('returnOnAssets'),
            "ROIC": info.get('returnOnCapital'),
            "Gross Margin": info.get('grossMargins'),
            "Operating Margin": info.get('operatingMargins'),
            "Net Profit Margin": info.get('profitMargins'),
            "Dividend Yield": info.get('dividendYield'),
            "Payout Ratio": info.get('payoutRatio'),
        }

        # Growth Rates
        revenue = financials.loc['Total Revenue']
        net_income = financials.loc['Net Income']
        revenue_growth = revenue.pct_change(periods=-1).iloc[0] if len(revenue) > 1 else None
        net_income_growth = net_income.pct_change(periods=-1).iloc[0] if len(net_income) > 1 else None

        growth_rates = {
            "Revenue Growth (YoY)": revenue_growth,
            "Net Income Growth (YoY)": net_income_growth,
        }

        # Valuation
        market_cap = info.get('marketCap')
        enterprise_value = info.get('enterpriseValue')

        valuation = {
            "Market Cap": market_cap,
            "Enterprise Value": enterprise_value,
            "EV/EBITDA": info.get('enterpriseToEbitda'),
            "EV/Revenue": info.get('enterpriseToRevenue'),
        }

        # Future Estimates
        estimates = {
            "Next Year EPS Estimate": info.get('forwardEps'),
            "Next Year Revenue Estimate": info.get('revenueEstimates', {}).get('avg'),
            "Long-term Growth Rate": info.get('longTermPotentialGrowthRate'),
        }

        # Simple DCF Valuation (very basic)
        free_cash_flow = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else None
        wacc = 0.1  # Assumed Weighted Average Cost of Capital
        growth_rate = info.get('longTermPotentialGrowthRate', 0.03)
        
        def simple_dcf(fcf, growth_rate, wacc, years=5):
            if fcf is None or growth_rate is None:
                return None
            terminal_value = fcf * (1 + growth_rate) / (wacc - growth_rate)
            dcf_value = sum([fcf * (1 + growth_rate) ** i / (1 + wacc) ** i for i in range(1, years + 1)])
            dcf_value += term
            
            inal_value / (1 + wacc) ** years
            return dcf_value

        dcf_value = simple_dcf(free_cash_flow, growth_rate, wacc)

        # Prepare the results
        analysis = {
            "Company Name": info.get('longName'),
            "Sector": info.get('sector'),
            "Industry": info.get('industry'),
            "Key Ratios": ratios,
            "Growth Rates": growth_rates,
            "Valuation Metrics": valuation,
            "Future Estimates": estimates,
            "Simple DCF Valuation": dcf_value,
            "Last Updated": datetime.fromtimestamp(info.get('lastFiscalYearEnd', 0)).strftime('%Y-%m-%d'),
            "Data Retrieval Date": datetime.now().strftime('%Y-%m-%d'),
        }

        # Add interpretations
        interpretations = {
            "P/E Ratio": "High P/E might indicate overvaluation or high growth expectations" if ratios.get('P/E Ratio', 0) > 20 else "Low P/E might indicate undervaluation or low growth expectations",
            "Debt to Equity": "High leverage" if ratios.get('Debt to Equity', 0) > 2 else "Conservative capital structure",
            "ROE": "Strong returns" if ratios.get('ROE', 0) > 0.15 else "Potential profitability issues",
            "Revenue Growth": "Strong growth" if growth_rates.get('Revenue Growth (YoY)', 0) > 0.1 else "Slowing growth",
        }

        analysis["Interpretations"] = interpretations

        return analysis

    except Exception as e:
        return f"An error occurred during the analysis: {str(e)}"



@tool
def yahoo_news_tool(stock_symbol):
    """
    Perform a comprehensive technical analysis on the given stock symbol.
    
    Args:
        stock_symbol (str): The stock symbol to analyze.
        period (str): The time period for analysis. Default is "1y" (1 year).
    
    Returns:
        dict: A dictionary with the detailed technical analysis results.
    """
    return YahooFinanceNewsInput(query=stock_symbol)


class InferlessPythonModel:
    def initialize(self):
        ollama_status = start_and_check_ollama()
        if ollama_status:
          self.llm =LLM(model="ollama/hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q4_K_L")
          self.agent = self.create_agent(self.llm)
        
    def infer(self, inputs):
        stock_symbol = inputs['stock_symbol']
        result = self.agent.kickoff(inputs={'stock_symbol': stock_symbol})
        result_str = str(result)
      
        return {"agent_response":result_str}

    def create_agent(self,llm):
        # Tools Initialization
        serper_tool = SerperDevTool()
        yf_fundamental_tool = yf_fundamental_analysis
        yf_tools = [LlamaIndexTool.from_tool(t) for t in YahooFinanceToolSpec().to_tool_list()]
        # Agents Definitions
        researcher = Agent(
            role='Equity Market Analyst',
            goal='Conduct an in-depth analysis of the financial performance and market position of {stock_symbol}',
            verbose=True,
            memory=True,
            backstory="Holding a Ph.D. in Financial Economics and possessing 15 years of experience in equity research, you are renowned for your meticulous data collection and insightful analysis. Your expertise encompasses evaluating financial statements, assessing market trends, and providing strategic investment recommendations.",
            tools=[serper_tool]+yf_tools,
            llm=llm,
            max_iter = 1,
            allow_delegation=True
        )
    
        fundamental_analyst = Agent(
            role='Senior Equity Fundamental Analyst',
            goal='Conduct a comprehensive fundamental analysis of {stock_symbol}, focusing on financial statements, valuation metrics, and key value drivers to assess intrinsic value.',
            verbose=True,
            memory=True,
            backstory="As a Chartered Financial Analyst (CFA) with 15 years of experience in value investing, you possess a deep understanding of financial statement analysis and valuation techniques. Your expertise includes identifying undervalued securities through meticulous examination of financial health, earnings quality, and market position.",
            tools=[yf_fundamental_tool],
            llm=llm,
            max_iter = 1,
            allow_delegation=True
        )
    
        reporter = Agent(
            role='Senior Investment Advisor',
            goal='Provide comprehensive stock analyses and strategic investment recommendations to impress a high-profile client.',
            verbose=True,
            memory=True,
            backstory="As the most experienced investment advisor, you integrate various analytical insights to formulate strategic investment advice. Currently, you are working for a highly important client whom you need to impress.",
            tools=[serper_tool, yf_fundamental_tool]+yf_tools,
            llm=llm,
            max_iter = 1,
            allow_delegation=False
        )
    
        # Task Definitions
        research_task = Task(
            description=
            (
                "Conduct research on {stock_symbol}. Your analysis should include:\n"
                "1. Current stock price and historical performance (5 years).\n"
                "2. Key financial metrics (P/E, EPS growth, revenue growth, margins) compared to industry averages and competitors.\n"
                "3. Recent news and press releases (past 1 month) and their potential impact.\n"
                "4. Analyst ratings and price targets (min 5 analysts), including consensus and notable insights.\n"
                "5. Reddit and social media sentiment analysis (100 posts), categorizing sentiments.\n"
                "6. Major institutional holders and recent changes.\n"
                "7. Competitive landscape and {stock_symbol}'s market share.\n"
                "8. Macro-economic and industry trends affecting the company.\n"
                "9. Regulatory and legal considerations.\n"
                "10. Environmental, Social, and Governance (ESG) factors compared with industry peers.\n"
                "Use reputable financial websites and databases for data. Include charts where applicable and cite all sources appropriately.\n"
            ),
            expected_output="A comprehensive 500-word research report covering all points with data sources, charts, and actionable insights.",
            agent=researcher
        )
        
        fundamental_analysis_task = Task(
            description=(
                "Conduct fundamental analysis of {stock_symbol}. Include:\n"
                "1. Review last 3 years of financial statements.\n"
                "2. Key ratios (P/E, P/B, P/S, PEG, Debt-to-Equity, etc.).\n"
                "3. Comparison with main competitors and industry averages.\n"
                "4. Revenue and earnings growth trends.\n"
                "5. Management effectiveness (ROE, capital allocation).\n"
                "6. Competitive advantages and market position.\n"
                "7. Growth catalysts and risks (2-3 years).\n"
                "8. DCF valuation model with assumptions.\n"
                "Use yf_fundamental_analysis tool for data."
            ),
            expected_output='A 100-word fundamental analysis report with buy/hold/sell recommendation and key metrics summary.',
            agent=fundamental_analyst
        )
    
        report_task = Task(
            description=(
                "Create an investment report on {stock_symbol}. Include:\n"
                "1. Executive Summary: Investment recommendation.\n"
                "2. Company Snapshot: Key facts.\n"
                "3. Financial Highlights: Top metrics and peer comparison.\n"
                "4. Fundamental Analysis: Top strengths and concerns.\n"
                "5. Risk and Opportunity: Major risk and growth catalyst.\n"
                "6. Investment Thesis: Bull and bear cases.\n"
                "7. Price Target: 12-month forecast.\n"
            ),
            expected_output='A 600-word investment report with clear sections mentioned in the description.',
            agent=reporter
        )
    
        # Crew Definition and Kickoff for Result
        self.crew = Crew(
            agents=[researcher, fundamental_analyst, reporter],
            tasks=[research_task, fundamental_analysis_task, report_task],
            process=Process.sequential,
            cache=True
        )
        return self.crew


    def finalize(self):
        pass
