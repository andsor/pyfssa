#!/bin/sh

while inotifywait -r -e modify -e create -e move -e delete --exclude '\.sw.?$' tests fss; do python setup.py test; done
