#!/bin/bash

cat $(dirname "$0")/../README.md |
sed -e 's,doc/images/,https://raw.githubusercontent.com/alugowski/matrepr/main/doc/images/,g' |
sed -e 's,(demo-,(https://nbviewer.org/github/alugowski/matrepr/blob/main/demo-,g'