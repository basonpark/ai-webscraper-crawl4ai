#!/bin/bash
# Script to remove sensitive data from Git history

echo "WARNING: This script will rewrite Git history. Ensure you've backed up important changes."
echo "Starting Git history cleanup..."

# Create a new orphan branch
git checkout --orphan temp_branch

# Add all files (except those gitignored)
git add .

# Commit
git commit -m "Initial commit after removing sensitive data"

# Delete the old main branch
git branch -D main

# Rename the temp branch to main
git branch -m main

# Force push to origin
echo "Cleanup complete. You can now force push with: git push -f origin main"
echo "OR run the following command to push immediately:"
echo "git push -f origin main" 