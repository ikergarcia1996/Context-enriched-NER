#!/bin/sh
mkdir -p genre_titles
mkdir -p genre_models
echo "Downloading mGENRE model"
cd genre_models && wget https://dl.fbaipublicfiles.com/GENRE/models.tar.gz && tar -xvzf models.tar.gz && cd ..
echo "Downloading TRIE and tittle2wikidataID"
cd genre_titles && wget http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_trie_with_redirect.pkl && wget http://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl && tar -xvzf titles.tar.gz && cd ..
echo "Please run the following commands:"
echo "cd fairseq && pip install --editable ./ && cd .."
echo "You need this specific version of fairseq to run GENRE"
