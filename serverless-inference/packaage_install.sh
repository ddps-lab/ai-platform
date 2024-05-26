#!/bin/bash

# install python3.9
sudo yum -y groupinstall "Development Tools"
sudo yum -y install openssl-devel bzip2-devel libffi-devel
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz
tar xvf Python-3.9.0.tgz
cd Python-3.9*/
./configure --enable-optimizations
sudo make altinstall

# install Package Setting
sudo yum -y install gcc-c++
sudo yum -y install python3-devel

# install Packages
mkdir /home/ec2-user/mountpoint/efs/packages
pip3.9 install --upgrade pip --user
pip3.9 install --upgrade --target /home/ec2-user/mountpoint/efs/packages/ numpy
pip3.9 install --upgrade --target /home/ec2-user/mountpoint/efs/packages/ Pillow==9.5.0
pip3.9 install --upgrade --target /home/ec2-user/mountpoint/efs/packages/ requests-toolbelt
pip3.9 install --upgrade --target /home/ec2-user/mountpoint/efs/packages/ tensorflow-cpu==2.15.0
