#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install "transformers==4.38.2" \
            "timm==0.6.13" \
            "gcsfs==2024.12.0" \
            "datasets[vision]" "pytorch-lightning>=1.6.4" \
            "nltk" \
            "sentencepiece" \
            "zss" \
            "sconf>=0.2.3" \
            "datasets" \
            "sentence-transformers==2.2.2"

git config --global user.email "miquelgc2003@gmail.com"
git config --global user.name "Miquel Gómez Corral"


git clone git@bitbucket.org:solverrepos/module_ocr_llm.git
git clone https://github.com/clovaai/donut.git

echo "✅ Environment setup complete!"