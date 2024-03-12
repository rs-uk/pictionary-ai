#!/bin/bash

# Prompt the user with a question
echo "Are you sure you want to erase all local processed data? (yes/no)"
read answer

# Check the user"s response
if [ "$answer" = "yes" ]; then
    echo "Deleting processed data..."
    rm -rf raw_data/processed_data
elif [ "$answer" = "no" ]; then
    echo "Aborting..."
else
    echo "Invalid response. Exiting..."
fi
