#!/bin/bash
# public ip
line=$(cat launcher/instancesInfo.txt | head -n 1)
hosts=($line)
while sleep 2; do
  for (( i=0; i<${#hosts[@]}; i++))
      do
      result=$(gcloud compute ssh spotvm${i} --command="curl -s \"http://metadata.google.internal/computeMetadata/v1/instance/preempted\" -H \"Metadata-Flavor: Google\"")
      if [[ "$result" -eq "FALSE" ]] ; then
        echo "${hosts[i]} Not Interrupted"
      else
        echo "${hosts[i]} Interrupted"
      fi
      done
done


