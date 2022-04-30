#!/usr/bin/env bash

shell_dir=$(dirname "$0")

cd "$shell_dir"

rm -rf ml/__pycache__

rm -f ai-lab-4_zz2960.zip

zip -r ai-lab-4_zz2960.zip ml learn README.md requirements.txt
