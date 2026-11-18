#!/bin/sh
set -eu
python=$1
shift
exec "$python" -m segregation.query "$@"
