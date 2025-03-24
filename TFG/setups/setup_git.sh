#!/bin/bash

git config --global user.email "miquelgc2003@gmail.com"
git config --global user.name "Miquel Gómez Corral"

git clone git@bitbucket.org:solverrepos/module_ocr_llm.git
git clone https://github.com/clovaai/donut.git

echo "✅ Environment setup complete!"