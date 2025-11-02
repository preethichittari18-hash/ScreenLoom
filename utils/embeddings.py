import torch
from transformers import Pipeline
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification
from utils.utils import logger

class FinBERTPipeline(Pipeline):
    """Custom pipeline for FinBERT to extract sentiment labels, scores, and sentence embeddings."""

    def _sanitize_parameters(self, **kwargs) -> tuple:
        """Sanitizes and extracts parameters for preprocessing.

        Args:
            **kwargs: Arbitrary keyword arguments, including 'text' for input.

        Returns:
            tuple: Preprocessing kwargs, forward kwargs, and postprocessing kwargs.
        """
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs["text"] = kwargs["text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, sentence: str) -> dict[str, torch.Tensor]:
        """Tokenizes the input sentence for model processing.

        Args:
            sentence (str): Input text to tokenize.

        Returns:
            dict[str, torch.Tensor]: Tokenized inputs as tensors.
        """
        return self.tokenizer(sentence, return_tensors="pt")

    def _forward(self, inputs: dict[str, torch.Tensor]) -> any:
        """Runs the model forward pass on tokenized inputs.

        Args:
            inputs (dict[str, torch.Tensor]): Tokenized input tensors.

        Returns:
            any: Model outputs including logits and hidden states.
        """
        return self.model(**inputs, output_hidden_states=True)

    def postprocess(self, outputs: any) -> dict[str, any]:
        """Processes model outputs into a structured result.

        Args:
            outputs (any): Raw model outputs with logits and hidden states.

        Returns:
            dict[str, any]: Dictionary with label, score, and embedding.
        """
        sentence_embedding = torch.mean(outputs.hidden_states[-1][0], dim=0).numpy()
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction_max_index = int(torch.argmax(predictions))
        label = self.model.config.id2label[prediction_max_index]
        return {
            'label': label,
            'score': predictions[0][prediction_max_index].item(),
            'embedding': sentence_embedding
        }


def embeds(input_val: str, limit: int) -> any:
    """Generates sentence embeddings using a FinBERT pipeline.

    Args:
        input_val (str): Input text to generate embeddings for.
        limit (int): Maximum length of input text to process.

    Returns:
        any: Sentence embedding as a numpy array.

    Raises:
        Exception: If pipeline initialization fails, registers and retries.
    """
    logger.debug("Attempting to initialize FinBERT pipeline for input of length %d.",
                 len(input_val))
    try:
        pipe = pipeline('finbert-pipeline-with-sentence-embedding',
                        model='ProsusAI/finbert', device=0)
    except Exception as e:
        logger.warning("Pipeline initialization failed: %s. Registering custom pipeline.", e)
        PIPELINE_REGISTRY.register_pipeline(
            'finbert-pipeline-with-sentence-embedding',
            pipeline_class=FinBERTPipeline,
            pt_model=AutoModelForSequenceClassification,
        )
        pipe = pipeline('finbert-pipeline-with-sentence-embedding',
                        model='ProsusAI/finbert', device=0)

    truncated_input = input_val[:min(len(input_val), limit)]
    logger.info("Generating embedding for truncated input of length %d.", len(truncated_input))
    outputs = pipe(truncated_input)
    return outputs['embedding']
