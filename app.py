from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from crewai_tools import LlamaIndexTool
from ollama_utils import start_and_check_ollama
from agent_tools import yf_fundamental_analysis, yahoo_news_tool


class InferlessPythonModel:
    def initialize(self):
        model_name = "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q3_K_L"
        ollama_status = start_and_check_ollama(model_name)
        if ollama_status:
          self.llm =LLM(model=model_name)
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
        self.llm = None
