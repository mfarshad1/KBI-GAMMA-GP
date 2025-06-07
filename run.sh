#!/bin/bash

# Iterate over subfolders
for dir in x1*; do
    # Navigate to the subfolder
    cd "$dir" || { echo "Failed to change directory to $dir"; continue; }
    # Remove file
    rm *.o*
    rm *init*
    rm *T*
    rm *0*
    rm *sed*
    rm *dump
    rm *FORCED
    rm *restart*

    # Submit job from within the subfolder
    if [[ "$dir" == "x1=1.0" ]]; then
        job_script="../batch_lj_mix_pure1"
    elif [[ "$dir" == "x1=0.0" ]]; then
        job_script="../batch_lj_mix_pure2"
    else
        job_script="../batch_lj_mix"
    fi
    qsub "$job_script"

    # Move back to the parent directory
    cd ..
done

