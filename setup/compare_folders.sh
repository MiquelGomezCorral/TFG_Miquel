# #!/bin/bash

# # Usage: ./compare_folders.sh /path/to/folder1 /path/to/folder2

FOLDER1="$1"
FOLDER2="$2"

echo "Comparing contents of:"
echo " - $FOLDER1"
echo " - $FOLDER2"
echo ""

# Loop through all files in FOLDER1
find "$FOLDER1" -type f | while read -r file1; do
    rel_path="${file1#$FOLDER1/}"
    file2="$FOLDER2/$rel_path"

    if [ -f "$file2" ]; then
        if ! cmp -s "$file1" "$file2"; then
            echo "MODIFIED: $rel_path"
            echo "Differences:"
            diff -u "$file1" "$file2"
            echo ""
        fi
    else
        echo "ONLY IN $FOLDER1/: $rel_path"
    fi
done

# Check for files in FOLDER2 that aren't in FOLDER1
find "$FOLDER2" -type f | while read -r file2; do
    rel_path="${file2#$FOLDER2/}"
    file1="$FOLDER1/$rel_path"

    if [ ! -f "$file1" ]; then
        echo "ONLY IN $FOLDER2/: $rel_path"
    fi
done
