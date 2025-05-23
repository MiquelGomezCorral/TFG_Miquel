#!/bin/bash

COMMIT_MESSAGE=""

# Parse arguments
while getopts "m:" opt; do
    case ${opt} in
        m)
            COMMIT_MESSAGE="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Error: -m requires a commit message" >&2
            exit 1
            ;;
    esac
done

# Ensure a commit message was provided
if [ -z "$COMMIT_MESSAGE" ]; then
    echo "Error: You must provide a commit message with -m"
    echo "Usage: $0 -m \"Your commit message\""
    exit 1
fi

# Git commit and push
git add .
if ! git commit -m "$COMMIT_MESSAGE"; then
    echo "❌ Commit failed."
    exit 1
fi

if ! git push; then
    echo "❌ Push failed."
    exit 1
fi

echo "✅ Changes committed and pushed with message: \"$COMMIT_MESSAGE\""
