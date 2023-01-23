#!/bin/bash

# Move to the current working directory
cd $CWD

# Activate the virtual environment
#mamba activate mAnDE

# Run the experiment with the provided arguments
java -jar $FILE $PBS_ARRAY_INDEX $PARAMS