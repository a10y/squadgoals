FROM tensorflow/tensorflow:2.21.0-gpu
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen
RUN uv run python -m nltk.downloader --dir=/usr/local/share/nltk_data perluniprops punkt punkt_tab
