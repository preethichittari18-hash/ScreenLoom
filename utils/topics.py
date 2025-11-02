from .embeddings import embeds
from .utils import TopicModel, logger, time_execution

import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class TopicExtractor:
    """Extracts topics from text data using embeddings, clustering, and LLM-based topic generation.

    Attributes:
        system_prompt (str): Prompt for the LLM defining its role in topic generation.
        few_shot_examples (list): Example inputs and outputs for few-shot learning with the LLM.
    """
    # TODO: Load models as part of class instead of object

    system_prompt = "You are an intelligent assistant that generates topics from text. Given a piece of text, you will generate a list of 3-4 relevant topics (don't use short form for anything) that summarize the main points or themes of the text, in the form of a structured list."

    few_shot_examples = [
        {
            "input": "The rapid advancement of artificial intelligence is transforming industries and creating new opportunities for innovation. AI technologies are being integrated into various sectors, including healthcare, finance, and transportation.",
            "output": ["Artificial Intelligence-driven Transformation", "Industry Transformation", "Innovation in Industry", "AI Integration in various Sectors"]
        },
        {
            "input": "Climate change is one of the most pressing issues of our time. Rising global temperatures, melting ice caps, and extreme weather events are just a few of the consequences of a warming planet. Efforts to mitigate climate change include reducing greenhouse gas emissions and transitioning to renewable energy sources.",
            "output": ["Climate Change Dangers", "Mitigating Climate Change", "Transition to Renewable Energy"]
        }
    ]

    @time_execution
    def __init__(self, data: pd.DataFrame, selected_column: str, status_bar=None, config=None):
        """Initializes the TopicExtractor with a DataFrame and selected column.

        Args:
            data (pd.DataFrame): Input DataFrame containing the text data.
            selected_column (str): Name of the column containing text to analyze.
            status_bar: Optional Streamlit progress bar object for status updates.
        """
        self.df = pd.DataFrame({"text": data[selected_column]})
        self.status_bar = status_bar
        self.EMBEDDER_CHARACTER_LIMIT = config.get("EMBEDDER_CHARACTER_LIMIT", 255) if config else 255


    @time_execution
    def _create_embeddings(self):
        """Creates embeddings for the text data in the DataFrame."""
        self.df['embeddings'] = self.df['text'].apply(lambda x: embeds(x, self.EMBEDDER_CHARACTER_LIMIT))

    @time_execution
    def _reduce_embedings(self, n_components: int):
        """Reduces the dimensionality of embeddings using UMAP.

        Args:
            n_components (int): Number of dimensions to reduce embeddings to.
        """
        embeddings: np.ndarray = np.array(self.df['embeddings'].tolist())
        reducer = UMAP(n_components=n_components)
        reduced_embeddings: np.ndarray = reducer.fit_transform(embeddings)
        self.df['embeddings'] = list(reduced_embeddings)

    @time_execution
    def _cluster(self, min_cluster_size: int, min_samples: int):
        """Clusters reduced embeddings using HDBSCAN.

        Args:
            min_cluster_size (int): Minimum size of clusters.
            min_samples (int): Minimum number of samples in a neighborhood for a point to be a core point.
        """
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        results = clusterer.fit(self.df['embeddings'].to_list())
        self.df['cluster'] = results.labels_

    @time_execution
    def _create_topics(self, api_key, input_limit: int):
        """Generates topics from clustered text using an LLM with few-shot prompting.

        Args:
            api_key (str): API key for the Grok LLM service.
            input_limit (int): Maximum character limit for text input to the LLM.

        Raises:
            Exception: If LLM invocation fails, logs the error and skips the cluster.
        """
        few_shot_prompt = FewShotPromptTemplate(
            examples=self.few_shot_examples,
            example_prompt=PromptTemplate(
                input_variables=["input", "output"],
                template="Text: {input}\nTopics: {output}"
            ),
            prefix=self.system_prompt,
            suffix="Text: {text}\nTopics: Only write the list of results and nothing else",
            input_variables=["text"],
            example_separator="\n\n"
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        structured_llm = llm.with_structured_output(TopicModel)

        chain = few_shot_prompt | structured_llm

        topics: list = []
        for i in range(0, self.df['cluster'].max() + 1):
            text_list = self.df['text'][self.df['cluster'] == i].to_list()
            texts = "\n".join(text_list)[:input_limit]
            try:
                result: TopicModel = chain.invoke({"text": texts})
                topics_list = result.topics_list
                topics.extend(topics_list)
            except Exception as e:
                print(f"Error: {e}")

        self.topics = pd.DataFrame({"Topic Name": topics})

    @time_execution
    def fit_transform(self,
                      llm_api_key: str,
                      reducer_n_components: int = 15,
                      clusterer_min_cluster_size: int = 5,
                      clusterer_min_samples: int = 5,
                      llm_input_limit: int = 5000,
                      ) -> pd.DataFrame:
        """Fits the topic extraction model and transforms the data into a topics DataFrame.

        Args:
            llm_api_key (str): API key for the Grok LLM service.
            reducer_n_components (int, optional): Number of components for UMAP reduction. Defaults to 15.
            clusterer_min_cluster_size (int, optional): Minimum cluster size for HDBSCAN. Defaults to 5.
            clusterer_min_samples (int, optional): Minimum samples for HDBSCAN. Defaults to 5.
            llm_input_limit (int, optional): Maximum character limit for LLM input. Defaults to 5000.

        Returns:
            pd.DataFrame: DataFrame containing extracted topics.
        """
        self._create_embeddings()
        if self.status_bar:
            self.status_bar.progress(40, text="Extracting topics...")
        self._reduce_embedings(n_components=reducer_n_components)
        if self.status_bar:
            self.status_bar.progress(60, text="Extracting topics...")
        self._cluster(min_cluster_size=clusterer_min_cluster_size, min_samples=clusterer_min_samples)
        if self.status_bar:
            self.status_bar.progress(80, text="Extracting topics...")
        self._create_topics(input_limit=llm_input_limit, api_key=llm_api_key)
        if self.status_bar:
            self.status_bar.progress(100, text="Extracting topics...")

        return self.topics


class TopicReducer:
    """Reduces the number of topics by clustering similar topics using embeddings."""

    @time_execution
    def __init__(self, topics_df: pd.DataFrame, status_bar=None):
        """Initializes the TopicReducer with a DataFrame of topics.

        Args:
            topics_df (pd.DataFrame): DataFrame containing topics to reduce.
            status_bar: Optional Streamlit progress bar object for status updates.
        """
        self.df = topics_df
        self.status_bar = status_bar

    @time_execution
    def _create_embeddings(self):
        """Creates embeddings for the topic names in the DataFrame."""
        self.df['embeddings'] = self.df['Topic Name'].apply(embeds)

    @time_execution
    def _cluster(self, min_samples: int, eps: int):
        """Clusters topic embeddings using DBSCAN.

        Args:
            min_samples (int): Minimum number of samples in a neighborhood for a point to be a core point.
            eps (int): Maximum distance between two samples for them to be considered in the same neighborhood.
        """
        embeddings = np.array(self.df['embeddings'].tolist())
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        cluster_labels = clusterer.fit_predict(embeddings_scaled)
        self.df['cluster'] = cluster_labels

    @time_execution
    def _extract_unique(self):
        """Extracts unique topics by selecting one representative per cluster."""
        self.new_topics_df = self.df.drop_duplicates(subset=['cluster'])[['Topic Name']]

    @time_execution
    def fit_transform(self,
                      min_samples: int = 1,
                      eps: float = 20.0) -> pd.DataFrame:
        """Fits the topic reduction model and transforms the data into a reduced topics DataFrame.

        Args:
            min_samples (int, optional): Minimum samples for DBSCAN clustering. Defaults to 1.
            eps (float, optional): Maximum distance for DBSCAN clustering. Defaults to 20.0.

        Returns:
            pd.DataFrame: DataFrame containing reduced topics.
        """
        self._create_embeddings()
        if self.status_bar:
            self.status_bar.progress(40, text="Reducing topics...")
        self._cluster(min_samples=min_samples, eps=eps)
        if self.status_bar:
            self.status_bar.progress(80, text="Reducing topics...")
        self._extract_unique()
        if self.status_bar:
            self.status_bar.progress(100, text="Reducing topics...")

        return self.new_topics_df