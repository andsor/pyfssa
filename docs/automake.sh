#!/bin/sh

while inotifywait -qq -r -e modify -e create -e move -e delete \
       	--exclude '\.sw.?$|^_build' . ../fss
do
	clear
	make html
	sleep 1
done 
