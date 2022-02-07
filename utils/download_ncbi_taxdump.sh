#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKDIR="$BASEDIR/../ncbi_taxdump"
TARGET="ncbi_names.txt.gz"


function rename-to-bak () {
    local filename="$1"
    if [[ -f "$filename" ]]; then
        echo "[info] Renaming existing $filename to ${filename}.bak"
        mv -f "$filename" "${filename}.bak"
    fi
}


mkdir -p "$WORKDIR"
cd "$WORKDIR"

rm -f *.dmp
rename-to-bak taxdump.tar.gz
wget ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
tar -xf taxdump.tar.gz

rename-to-bak "$TARGET"
cat names.dmp | awk --field-separator '\t|\t' '{ print $3 }' | sed -r 's/^\s+//g' | sed -r 's/\s+$//g' | grep '\s' | gzip -c > "$TARGET"

if [[ -f "$TARGET" ]]; then
    echo "[info] Successfully saved as ncbi_names.txt.gz"
else
    echo "[info] Failed to download and process!"
fi
