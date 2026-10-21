#!/usr/bin/env bash
# seal.sh OUT.enc FILE... : pack files, encrypt with a fresh random AES-256 key, wrap that key with the
# owner's RSA public key (keys/out_public.pem). Only the holder of the private key can open it.
set -euo pipefail
out=$1; shift
here=$(cd "$(dirname "$0")/.." && pwd)
K=$(openssl rand -hex 32); export K
tar -czf - "$@" | openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -pass env:K -out "$out.data"
printf %s "$K" | openssl pkeyutl -encrypt -pubin -inkey "$here/keys/out_public.pem" -pkeyopt rsa_padding_mode:oaep -out "$out.key"
tar -cf "$out" -C "$(dirname "$out")" "$(basename "$out").data" "$(basename "$out").key"
rm -f "$out.data" "$out.key"; unset K
