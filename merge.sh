#!/bin/bash

find . -type f ! -path "./.git/*" | while read -r file; do
  if grep -q "<<<<<<< HEAD" "$file"; then
    awk '
      BEGIN {keep=0}
      /^<<<<<<< HEAD$/ {keep=0; next}
      /^=======$/ {keep=1; next}
      /^>>>>>>> / {keep=0; next}
      keep {print}
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    echo "Resolved: $file"
  fi
done
