#!/bin/bash

FROM="$1"
TO="$2"

while :
do
    gsutil -m rsync -r "$FROM" "$TO"
    sleep 60
done

