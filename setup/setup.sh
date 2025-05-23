#!/bin/bash
# set -e

# Parse arguments
ALL=false
PIP=false
GIT=false
YES=false

while getopts "apgy" opt; do
  case ${opt} in
    a ) ALL=true ;;
    p ) PIP=true ;;
    g ) GIT=true ;;
    y ) YES=true ;;
    * ) echo "Usage: $0 [-p] (pip install) [-g] (git clone/pull) [-a] (do both) [-y] (accept all, pulling)"; exit 1 ;;
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
    if [ -f "./scripts_environment/requirements.txt" ]; then
        pip install -r "./scripts_environment/requirements.txt"
    else
        echo "❌ Error: requirements.txt not found at ./scripts_environment/requirements.txt"
        exit 1
    fi
fi 

if [ "$GIT" = true ]; then
    echo -e "\n======================================"
    echo "         PULLING SELF (TFG_Miquel)"
    echo "======================================"

    git config --global user.email "miquelgc2003@gmail.com"
    git config --global user.name "Miquel Gómez Corral"

    if [ "$YES" = true ]; then
        git pull
    else
      read -p "Do you want to pull the latest changes? (y/n): " answer

      if [[ "$answer" =~ ^[Yy]$ ]]; then
          git pull
      else
          echo "Skipping git pull."
      fi
    fi
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

