#coding:utf-8
import json
import subprocess
import time
from profiler import instanceinfo

image_id = instanceinfo.image_id
subnet_id = instanceinfo.subnet_id
SecurityGroupIds = instanceinfo.SecurityGroupIds
key = instanceinfo.key

def getSpotInstance(instance_type, count):
    instance_id = ""
    hosts = []
    public_ips = []
    private_ips = [] 
    batchs = []
    instance_type_index= []
    
    print("\n****************** apply for spot instances ********************")

    for i in range(len(instance_type)):
        with open("launcher/specification.json.template", 'r') as f1, open("launcher/instanceSpec.json", 'w') as f2:
            template = f1.read()
            specification_file = template % (image_id, key, SecurityGroupIds, instance_type[i], subnet_id)
            f2.write(specification_file)
            
        if count[i] == 0:
            continue 
        print("\n %d %s instances ... " %(count[i], instance_type[i]))
        command = """aws ec2 request-spot-instances --instance-count %d \
                                                    --type one-time \
                                                    --launch-specification file://launcher/instanceSpec.json""" %(count[i])
        
        spot_instance_request_id = ""
        try:
            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)
            for j in range(count[i]):
                rid = return_obj["SpotInstanceRequests"][j]["SpotInstanceRequestId"] 
                spot_instance_request_id = spot_instance_request_id + ' ' + rid
            # print(" spot_instance_request_id: ",spot_instance_request_id)
            command = """aws ec2 describe-spot-instance-requests --spot-instance-request-id %s""" % (spot_instance_request_id)
            time.sleep(10)
            
            try:
                output = subprocess.check_output(command, shell=True).decode()
                return_obj = json.loads(output)
                instanceid_one_type = ""
                for j in range(count[i]):
                    id = return_obj["SpotInstanceRequests"][j]["InstanceId"]
                    instance_id = instance_id + ' ' + id
                    instanceid_one_type = instanceid_one_type + ' ' + id
                
            except Exception as e:
                print("\n instance id not enough !")
                
            if instanceid_one_type is not None:
                command = """aws ec2 describe-instances --instance-ids %s \
                                                    --query "Reservations[*].Instances[*].{hosts:PublicIpAddress,hosts_private:PrivateIpAddress}" \
                                                    --output json""" %(instanceid_one_type)

                try:
                    output = subprocess.check_output(command, shell=True).decode()
                    return_obj = json.loads(output)
                    for j in range(count[i]):
                        public_ips.append(return_obj[0][j]["hosts"])
                        private_ips.append(return_obj[0][j]["hosts_private"])
                        batchs.append(instanceinfo.instance_batch[i])
                        instance_type_index.append(i)
                    
                except Exception as e:
                    print("\n Failed to get ip address!")  
                
        except Exception as e:
            print("\n Failed to get request id !")
    
        if spot_instance_request_id is not None:
            command = """aws ec2 cancel-spot-instance-requests --spot-instance-request-ids %s""" % (spot_instance_request_id)
            subprocess.check_output(command, shell=True)
    
    hosts.append(public_ips)
    hosts.append(private_ips)
    hosts.append(batchs)
    hosts.append(instance_type_index)
    
    print("\n----------------- result -----------------")
    print("\npublic_ips:", hosts[0])
    print("\nprivate_ips:", hosts[1])
    print("\ninstance_ids:", instance_id)

    print("\n******************** WRITE INSTANCES INFO INTO FILE !!*********************\n")

    with open("launcher/instancesInfo.txt", "w") as f:
        for item in hosts[0]:
            f.write(item +' ')
        f.write('\n')
        for item in hosts[1]:
            f.write(item +' ')
        f.write('\n')
        for item in hosts[2]:
            f.write(str(item) +' ')
        f.write('\n')
        for item in hosts[3]:
            f.write(str(item) +' ')
        f.write('\n')
        f.write(instance_id)
    
    return hosts