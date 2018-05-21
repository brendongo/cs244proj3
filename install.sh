
# Install mininet
cd ~/
git clone git://github.com/mininet/mininet
cd mininet
git checkout -b 2.0.0
cd ..
mininet/util/install.sh -a

git clone https://github.com/brendongo/cs244proj2.git
cd cs244proj2

sudo apt-get -y install python-setuptools python-dev build-essential 
sudo easy_install pip 

sudo pip install -r requirements.txt

mkdir results/
