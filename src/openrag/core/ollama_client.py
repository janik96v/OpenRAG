"""Ollama client for LLM-based context generation."""

import asyncio
import logging
from typing import Optional

import ollama

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Exception raised when Ollama operations fail."""

    pass


class OllamaClient:
    """Client for interacting with Ollama API with retry logic and timeouts."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API
            timeout: Timeout in seconds for API calls (default: 300s / 5 minutes)
            max_retries: Maximum number of retry attempts on failure
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = ollama.AsyncClient(host=base_url)

        logger.info(f"Initialized OllamaClient with base_url={base_url}, timeout={timeout}s")

    async def generate_response(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> str:
        """
        Generate response from Ollama with exponential backoff retry.

        Args:
            prompt: User prompt for generation
            model: Model to use (e.g., 'llama3.2:3b')
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            OllamaError: If generation fails after all retries
        """
        attempt = 0
        last_exception = None

        while attempt < self.max_retries:
            try:
                logger.debug(
                    f"Generating response with model={model}, attempt={attempt + 1}/{self.max_retries}"
                )

                # Build messages
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                # Call Ollama API with timeout
                response = await asyncio.wait_for(
                    self.client.chat(
                        model=model,
                        messages=messages,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    ),
                    timeout=self.timeout,
                )

                # Extract response text
                generated_text = response["message"]["content"]

                logger.debug(f"Successfully generated response ({len(generated_text)} chars)")

                return generated_text

            except (ConnectionError, asyncio.TimeoutError, Exception) as e:
                last_exception = e
                attempt += 1

                if attempt < self.max_retries:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2**attempt
                    logger.warning(
                        f"Generation failed (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Generation failed after {self.max_retries} attempts: {e}")

        # All retries exhausted
        raise OllamaError(
            f"Failed to generate response after {self.max_retries} attempts: {last_exception}"
        )

    def model_exists(self, model: str) -> bool:
        """
        Check if a model is available locally.

        Args:
            model: Model name to check

        Returns:
            True if model exists, False otherwise
        """
        try:
            models = ollama.list()
            model_names = [m["name"] for m in models.get("models", [])]
            return model in model_names
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return False

    def get_model_info(self, model: str) -> dict:
        """
        Get information about a specific model.

        Args:
            model: Model name

        Returns:
            Dictionary with model information

        Raises:
            OllamaError: If model info retrieval fails
        """
        try:
            info = ollama.show(model)
            return info
        except Exception as e:
            raise OllamaError(f"Failed to get model info for '{model}': {e}")
