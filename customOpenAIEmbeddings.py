from config import *

class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate length-safe embeddings for a list of texts.

        This method handles tokenization and embedding generation, respecting the
        set embedding context length and chunk size. It supports both tiktoken
        and HuggingFace tokenizer based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        _chunk_size = chunk_size or self.chunk_size

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer  # noqa: F401
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed in order to for OpenAIEmbeddings without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk = tokenized[j : j + self.embedding_ctx_length]

                    # Convert token IDs back to a string
                    chunk_text = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            # try:
            #     encoding = tiktoken.encoding_for_model(model_name)
            # except KeyError:
            #     encoding = tiktoken.get_encoding("cl100k_base")
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                # token = encoding.encode(
                #     text=text,
                #     allowed_special=self.allowed_special,
                #     disallowed_special=self.disallowed_special,
                # )

                token = text

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(tokens), _chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), _chunk_size)
        else:
            _iter = range(0, len(tokens), _chunk_size)

        batched_embeddings: List[List[float]] = []
        for i in _iter:
            response = self.client.create(
                input=tokens[i : i + _chunk_size], **self._invocation_params
            )
            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average_embedded = self.client.create(
                    input="", **self._invocation_params
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                average = average_embedded["data"][0]["embedding"]
            else:
                # should be same as
                # average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
                total_weight = sum(num_tokens_in_batch[i])
                average = [
                    sum(
                        val * weight
                        for val, weight in zip(embedding, num_tokens_in_batch[i])
                    )
                    / total_weight
                    for embedding in zip(*_result)
                ]

            # should be same as
            #  embeddings[i] = (average / np.linalg.norm(average)).tolist()
            magnitude = sum(val**2 for val in average) ** 0.5
            embeddings[i] = [val / magnitude for val in average]

        return embeddings

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Asynchronously generate length-safe embeddings for a list of texts.

        This method handles tokenization and asynchronous embedding generation,
        respecting the set embedding context length and chunk size. It supports both
        `tiktoken` and HuggingFace `tokenizer` based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        _chunk_size = chunk_size or self.chunk_size

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed in order to for OpenAIEmbeddings without "
                    " `tiktoken`. Please install it with `pip install transformers`."
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk = tokenized[j : j + self.embedding_ctx_length]

                    # Convert token IDs back to a string
                    chunk_text = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            # try:
            #     encoding = tiktoken.encoding_for_model(model_name)
            # except KeyError:
            #     logger.warning("Warning: model not found. Using cl100k_base encoding.")
            #     model = "cl100k_base"
            #     encoding = tiktoken.get_encoding(model)
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                # token = encoding.encode(
                #     text=text,
                #     allowed_special=self.allowed_special,
                #     disallowed_special=self.disallowed_special,
                # )

                token = text

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        batched_embeddings: List[List[float]] = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = await self.async_client.create(
                input=tokens[i : i + _chunk_size], **self._invocation_params
            )

            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average_embedded = await self.async_client.create(
                    input="", **self._invocation_params
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                average = average_embedded["data"][0]["embedding"]
            else:
                # should be same as
                # average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
                total_weight = sum(num_tokens_in_batch[i])
                average = [
                    sum(
                        val * weight
                        for val, weight in zip(embedding, num_tokens_in_batch[i])
                    )
                    / total_weight
                    for embedding in zip(*_result)
                ]
            # should be same as
            # embeddings[i] = (average / np.linalg.norm(average)).tolist()
            magnitude = sum(val**2 for val in average) ** 0.5
            embeddings[i] = [val / magnitude for val in average]

        return embeddings
