#!/usr/bin/env bash
# unseal.sh X.tar.enc PRIVATE.pem OUTDIR : open a sealed output (owner side only)
set -euo pipefail
d=$(mktemp -d); tar -xf "$1" -C "$d"; b=$(basename "$1")
K=$(openssl pkeyutl -decrypt -inkey "$2" -pkeyopt rsa_padding_mode:oaep -in "$d/$b.key"); export K
mkdir -p "$3"; openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:K -in "$d/$b.data" | tar -xzf - -C "$3"
rm -rf "$d"; unset K
