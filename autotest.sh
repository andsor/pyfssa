#!/bin/sh

while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' tests fss
do
	clear
	python -m unittest discover
	sleep 1
done 
