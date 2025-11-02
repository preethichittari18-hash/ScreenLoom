from .utils import logger, time_execution

import pandas as pd
import time

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Associator:
    """A class to analyze text documents and associate them with topics, sentiments, and issues."""

    system_prompt = """You are a general natural language model which comes up with the following 
    insights from the case that you are given. Each case will consist of a Document, a Key 
    Entity, and a list of Topics along with their numbered index. For the case, you must extract 
    the following information:
    1. Topic: What topic does the Document belong to? It should be the index value. Note that 
    it should not exceed the length of the Topics list.
    2. Sentiment: What is the emotion which the text is giving about the specified Entity? It 
    should be of the class ["NEGATIVE", "NEUTRAL", "POSITIVE"] only. Focus on the opinion of the 
    text on the particular Entity for this analysis. Return NEUTRAL ONLY IF the Entitity is not 
    mentioned in the document. If Entity is None, then return sentiment regarding the entire 
    document.
    4. Strength: What is the confidence with which you have assigned to sentiment? Try to be 
    objective and return a float only according to how strongly or mildly the sentiment is 
    presented in the text.
    3. Issues: What issues are highlighted in the text, especially concerning the Key Entity. 
    If Key Entity is None, return issues regarding the entire Document.
    You must return these according to the provided pydantic schema.

    Topics: {topics_dict}
    Document: {document}
    Key Entity: {key_entity}
    Provide the output as a JSON object with keys "topic_label", "sentiment", "strength", and "issues".
    """

    prompt = PromptTemplate(
        input_variables=["topics_dict", "document", "key_entity"],
        template=system_prompt
    )

    @time_execution
    def __init__(self, data: pd.DataFrame,
                 topics_df: pd.DataFrame,
                 selected_column: str,
                 entity_column: str,
                 status_bar = None):
        """
        Initialize the Associator with data and configuration.

        Args:
            data (pd.DataFrame): Input data containing documents
            topics_df (pd.DataFrame): DataFrame containing topic information
            selected_column (str): Column name containing the text documents
            entity_column (str): Column name containing the key entities
            status_bar: Optional progress bar object for tracking execution
        """
        self.df = data.iloc[:100]
        self.selected_column = selected_column
        self.entity_column = entity_column
        self.topics_df = topics_df
        self.topics_length = topics_df.shape[0] - 1
        self.topics_dict = dict(zip(range(0, topics_df.shape[0]), topics_df["Topic Name"]))
        self.status_bar = status_bar

    @time_execution
    def _load_dynamic_validator(self):
        """
        Create a dynamic Pydantic model for validating analysis results based on available topics.
        """
        topics_length = self.topics_length
        class CaseModel(BaseModel):
            topic_label: int = Field(..., description="Topic which the Document belongs to.")
            sentiment: Literal["NEGATIVE", "NEUTRAL", "POSITIVE"] = Field(default="NEUTRAL", description="Sentiment conveyed about the specified Entity. If no mention of that Entity, then NEUTRAL. If Entity is None, then sentiment of that article (in which case it shouldn't by NEUTRAL).")
            strength: float = Field(..., description="Strength of the sentiment provided regarding the specified Entity or Document between 0.0 to 1.0.")
            issues: list[str] = Field(description="Issues mentioned in the Document. Mention issues about Key Entity if it exists.")

            @field_validator("topic_label")
            @classmethod
            def validate_topic_label(cls, value: int) -> int:
                if not (0 <= value <= topics_length):
                    raise ValueError(f"topic_label must be between 0 and {topics_length}, got {value}")
                return value
            
        self.CaseModel = CaseModel

    @time_execution
    def _load_chain(self, api_key: str):
        """
        Initialize the language model chain with structured output.

        Args:
            api_key (str): API key for accessing the language model
        """
        llm = ChatGroq(model="gemma2-9b-it", api_key=api_key)
        structured_llm = llm.with_structured_output(self.CaseModel)

        self.chain = self.prompt | structured_llm

    @time_execution
    def _run_chain(self):
        """
        Process documents through the language model chain and collect results.
        """
        def invoke_with_retries(chain, input_data, max_retries=3, delay=1):
            """
            Attempt to invoke the chain with retries on failure.

            Args:
                chain: The language model chain to invoke
                input_data: Data to process
                max_retries (int): Maximum number of retry attempts
                delay (int): Delay between retries in seconds

            Returns:
                Response from the chain

            Raises:
                Exception: If all retries fail
            """
            for attempt in range(max_retries):
                try:
                    response = chain.invoke(input_data)
                    return response
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    continue
            raise Exception(f"Failed after {max_retries} retries: Unable to generate valid output")

        results = []
        for i in range(0, self.df.shape[0]):
            time.sleep(10)
            text = self.df.iloc[i][self.selected_column][:10000]
            drug_name = self.df.iloc[i][self.entity_column] if self.entity_column in self.df.columns else None
            input_data = { "topics_dict": self.topics_dict, "document": text, "key_entity": drug_name }
            try:
                result = invoke_with_retries(self.chain, input_data, max_retries=3, delay=1)
            except Exception as e:
                logger.error(f"Retries failed for document inference due to: {e}")
                result = {"topic_label": -1, "sentiment": "UNKNOWN", "entities": ["COULDN'T EXTRACT"]}
            results.append(result)
            if (self.status_bar): self.status_bar.progress(int(10 + i * (60 / self.df.shape[0])), text="Extracting features...")
        self.results = results

    @time_execution
    def _format_data(self):
        """
        Format the analysis results into a DataFrame.
        """
        topic_labels = [x.topic_label if isinstance(x, self.CaseModel) else None for x in self.results]
        topics = [self.topics_df['Topic Name'].iloc[label] if (label is not None) else "COULDN'T ALLOCATE" for label in topic_labels]
        strengths = [x.strength if isinstance(x, self.CaseModel) else None for x in self.results]
        sentiments = [x.sentiment if isinstance(x, self.CaseModel) else None for x in self.results]
        issues = [x.issues if isinstance(x, self.CaseModel) else None for x in self.results]

        results_df = pd.DataFrame({
            "topic": topics,
            "sentiment": sentiments,
            "strength": strengths,
            "issues": issues
        })
        self.results_df = results_df

    @time_execution
    def fit_transform(self, llm_api_key: str) -> pd.DataFrame:
        """
        Process the data and return analyzed results.

        Args:
            llm_api_key (str): API key for the language model

        Returns:
            pd.DataFrame: DataFrame containing original data plus analysis results
        """
        self._load_dynamic_validator()
        if (self.status_bar):
            self.status_bar.progress(5, text="Extracting features...")
        self._load_chain(api_key=llm_api_key)
        if (self.status_bar):
            self.status_bar.progress(10, text="Extracting features...")
        self._run_chain()
        if (self.status_bar):
            self.status_bar.progress(80, text="Extracting features...")
        self._format_data()
        if (self.status_bar):
            self.status_bar.progress(100, text="Extracting features...")

        final_df = pd.concat([self.df, self.results_df], axis=1)
        return final_df