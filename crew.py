import os
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

# Import Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Tools
search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
researcher = Agent(
    role='Tech Innovation Analyst',
    goal='Identify new blockchain technologies and related innovative project ideas using available resources.',
    backstory="""You are an expert in blockchain technologies and related innovations.
    Your work involves staying ahead of blockchain trends and evaluating new concepts for their potential impact and feasibility, given the constraints of available tools and data.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ChatGoogleGenerativeAI(model="gemini-pro")
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on blockchain advancements and innovative ideas that have potential.',
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex blockchain concepts into compelling narratives. You write only the project ideas.""",
    verbose=True,
    llm=ChatGoogleGenerativeAI(model="gemini-pro"),
    allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
    description="""Conduct a targeted analysis using available tools to identify significant blockchain technologies and innovative project ideas.
    Focus on trends and breakthroughs in the blockchain space that are visible through current resources and tools.""",
    expected_output="Summary of notable blockchain technologies and ideas based on available information.",
    llm=ChatGoogleGenerativeAI(model="gemini-pro"),
    agent=researcher
)

task2 = Task(
    description="""Based on the insights from task 1, develop a blog post highlighting the most significant blockchain technologies and ideas.
    Ensure the content is engaging and tailored to the insights gathered, even if limited.""",
    expected_output="Blog post summarizing key blockchain insights with at least 4 bullet points.",
    agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
try:
    result = crew.kickoff()
    print("######################")
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")
