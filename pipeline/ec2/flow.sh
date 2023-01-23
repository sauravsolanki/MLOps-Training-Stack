# update
sudo yum update -y
#// install most recent package
sudo amazon-linux-extras install docker
#// start the service docker
sudo service docker start
#// add the ec2-docker user to the group
sudo usermod -a -G docker ec2-user
#// you need to logout to take affect
#logout

sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose


#Try:
#
#Check the status:
#$ sudo service docker status
#If there isn’t running:
#$ sudo service docker start
#And then to auto-start after reboot:
#$ sudo systemctl enable docker
#Don’t forget to add your ec2-user to the docker group:
#$ sudo usermod -aG docker ec2-user
#And then reboot or just logoff/login again
#$ sudo reboot



#// login again
ssh -i "ec2-docker.pem" ec2-user@ec2-3-18-220-172.us-east-2.compute.amazonaws.com

docker-compose version

#// check the docker version
docker --version

# copy the pipeline folder
sudo scp -i ~/Downloads/mlops-level-4.pem -r pipeline/  ec2-user@ec2-13-235-16-226.ap-south-1.compute.amazonaws.com:~/.



# increase volums
#change security group

#cd pipeline; docker compose up
