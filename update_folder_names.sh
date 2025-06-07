#!/bin/bash

# Define the list of numbers
numbers=(0.048737372 0.189931184 0.335212726 0.469114793 0.603581663 0.75157274 0.843028938 0.934411407)

# Get a sorted list of folders starting with 'x1=' sorted from highest to lowest
folders=$(ls -d x1=* | sort -t '=' -k2,2nr)
# Get a sorted list of folders starting with 'x1=' sorted from lowest to highest
folders=$(ls -d x1=* | sort -t '=' -k2,2n)

# Initialize counter
counter=0

# Loop through each folder and rename
for folder in $folders; do
    # Extract the current number from the folder name
    current_number=$(echo $folder | cut -d '=' -f 2)

    # Get the new number from the array
    new_number=${numbers[$counter]}

    # Form the new folder name
    new_folder_name="x1=${new_number}"

    # Rename the folder
    mv "$folder" "$new_folder_name"

    # Increment the counter
    counter=$((counter + 1))
done

