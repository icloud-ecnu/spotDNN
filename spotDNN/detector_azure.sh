#!/bin/bash
# public ip
line=$(cat launcher/instancesInfo.txt | head -n 1)
hosts=($line)
while sleep 2; do
  for host in "${hosts[@]}"
      do
      cmd="curl -s -H Metadata:true http://169.254.169.254/metadata/scheduledevents?api-version=2019-08-01"
      result_json=$(ssh -o StrictHostKeyChecking=no -i ~/.ssh/Azure-Faye.pem azureuser@${host} ${cmd})

      if(($(echo $result_json | jq '.DocumentIncarnation') == 0)); then
        echo "${host} Not Interrupted"
      elif (($(echo $result_json | jq '.DocumentIncarnation') != 0)) && [[ $(echo $result_json | jq '.Events[0].EventType') == "\"Preempt\"" ]]; then
        echo "${host} Interrupted"
      else
        echo 'error'
      fi
      done
done

