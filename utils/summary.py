from .utils import time_execution

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


class SummaryGenerator:
    """Generates a summary and suggestions from topic-issue pairs using an LLM.

    Attributes:
        system_prompt (str): Prompt template for the LLM to generate summaries and suggestions.
        prompt (PromptTemplate): LangChain prompt object for structuring the input.
    """

    # system_prompt = """You are a general purpose text generation agent. For the given list
    #   of topics-issue pairs, you must write a summary/conclusion paragraph, using their 
    #   corresponding issues. First, you should highlight most impactful issues. Then, for each
    #   topic, you should provide overview issues and how they can be resolved. Do it from a
    #   recommendation perspective, as in what actions can be taken by the reader, assuming that they
    #   have the agency to carry out your advice. Make it concise and around a paragraph long. 
    # Issues List: {issues_list}
    # """
    system_prompt = """You are a general purpose text generation agent. For the given list
      of topics-issue pairs, you must write a summary/conclusion paragraph, using their 
      corresponding issues. It should highlight issues in the medicines and how their pharma
      companies can resolve them. Make it more like a recommendation paragraph. Keep it concise,
      and limited to a paragraph. 
    Issues List: {issues_list}
    """

    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["issues_list"]
    )

    def __init__(self, issues_list, status_bar):
        """Initializes the SummaryGenerator with topic-issue pairs and a status bar.

        Args:
            issues_list: List of tuples pairing topics/entities with their issues.
            status_bar: Streamlit progress bar object for status updates.
        """
        self.issues_list = issues_list
        self.status_bar = status_bar

    def _generate_summary(self, api_key: str):
        """Generates a summary from the issues list using an LLM.

        Args:
            api_key (str): API key for the Grok LLM service.
        """
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        chain = self.prompt | llm
        summary = chain.invoke({ "issues_list": self.issues_list })
        self.summary = summary

    @time_execution
    def fit_transform(self, api_key: str) -> str:
        """Fits the summary generator and transforms the issues list into a summary string.

        Args:
            api_key (str): API key for the Grok LLM service.

        Returns:
            str: Generated summary text including conclusions and suggestions.
        """
        if self.status_bar:
            self.status_bar.progress(50, text="Generating summary...")
        self._generate_summary(api_key=api_key)
        if self.status_bar:
            self.status_bar.progress(100, text="Generating summary...")
        return self.summary.content
