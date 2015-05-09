#!/bin/sh

while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' fssa
do
	clear
	py.test --cov=fssa fssa
	sleep 1
done
