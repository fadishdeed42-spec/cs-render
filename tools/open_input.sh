#!/usr/bin/env bash
# open_input.sh JOB.enc DIR : decrypt an input job with the CS_KEY repository secret and unpack into DIR
set -euo pipefail
mkdir -p "$2"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:CS_KEY -in "$1" | tar -xzf - -C "$2"
