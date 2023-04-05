#!/bin/bash
func_monitor(){
    echo $args
    read line < launcher/instancesInfo.txt
    hosts=($line)

    i=0
    for host in "${hosts[@]}"
    do
            echo "---- NOW MONITOR HOST ${i}: ${host} "

            command="mkdir -p /home/ubuntu/log && \
                        cd netmon/ && ./netmon.sh /home/ubuntu/log/bandwith_host${i} $args"
            echo $command
            ssh -i tf-faye.pem ubuntu@${host} "$command" &
            let i+=1
    done

    sleep $args
}
args=$1
func_monitor