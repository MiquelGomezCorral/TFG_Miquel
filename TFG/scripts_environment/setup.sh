#!/bin/bash
# set -e

# Parse arguments
ALL=false
PIP=false
GIT=false

while getopts "apg" opt; do
  case ${opt} in
    a ) ALL=true ;;
    p ) PIP=true ;;
    g ) GIT=true ;;
    * ) echo "Usage: $0 [-p] (pip install) [-g] (git clone/pull) [-a] (do both) "; exit 1 ;;
  esac
done

# If -a is passed, enable both PIP and GIT
if [ "$ALL" = true ]; then
  PIP=true
  GIT=true
fi



if [ "$PIP" = true ]; then
    echo -e "\n======================================"
    echo "           PIP INSTALLING"
    echo "======================================"
    # Upgrade pip
    pip install --upgrade pip


    # Install dependencies from requirements.txt
    if [ -f "./TFG/requirements.txt" ]; then
        pip install -r "./TFG/requirements.txt"
    else
        echo "❌ Error: requirements.txt not found at ./TFG/requirements.txt"
        exit 1
    fi


    # echo -e "\n======================================"
    # echo "       VS CODE EXTENSIONS CHECK"
    # echo "======================================"

    # # Locate VS Code binary dynamically
    # CODE_BIN=$(find ~/.vscode-server/bin -name "code" 2>/dev/null | head -n 1)

    # # Check if VS Code server is installed
    # if [ -z "$CODE_BIN" ]; then
    #     echo "❌ VS Code server not found. Make sure you're running this inside a VS Code-connected Colab environment."
    #     exit 1
    # fi

    # echo "✅ VS Code server found at: $CODE_BIN"

    # # List installed extensions
    # echo "🔍 Listing installed VS Code extensions..."
    # $CODE_BIN --list-extensions
fi 

if [ "$GIT" = true ]; then
    echo -e "\n======================================"
    echo "              CLONNING"
    echo "======================================"

    git config --global user.email "miquelgc2003@gmail.com"
    git config --global user.name "Miquel Gómez Corral"


    if [ -d './module_ocr_llm/' ]; then
        echo -e "\n- PULLING module_ocr_llm"

        cd ./module_ocr_llm/
        git pull
        cd ../
    else
        echo -e "\n- CLONNING module_ocr_llm"
        git clone git@bitbucket.org:solverrepos/module_ocr_llm.git
    fi


    # if [ -d './donut/' ]; then
    #     echo -e "\n- PULLING Donut"
    #     cd ./donut/
    #     git pull
    #     cd ../
    # else
    #     echo -e "\n- CLONING Donut"
    #     git clone https://github.com/clovaai/donut.git
    # fi
fi


if [[ "$ALL" = true || "$PIP" = true || "$GIT" = true ]]; then
  echo -e "\n======================================"
  echo "   ✅ Environment setup complete!"
  echo "======================================"
else
  echo ""
  echo "   ❌ You need to pass at least one tag"
  echo "Usage: $0 [-p] (pip install) [-g] (git clone/pull) [-a] (do both) "
fi

