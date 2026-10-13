FROM tensorflow/tensorflow:2.21.0-gpu
RUN pip install nltk tqdm six matplotlib pytest
RUN python -m nltk.downloader --dir=/usr/local/share/nltk_data perluniprops punkt
