# Upload kaggle.json before launching the script
# Modify directory if necessary

# data_dir example: ~/data/airbus_ship_detection
# Token kaggle dir example: ~/kaggle.json

echo "install kaggle" 
sudo pip install kaggle --upgrade

cd ~/
mkdir .kaggle
ls -a

echo "
------------------------------------"
echo "Copying kaggle token" 
cp $2 .kaggle/

echo "
------------------------------------"
echo "Make sure kagle.json is in here"
cd .kaggle/
pwd 
ls 
cd ..
chmod 600 .kaggle/kaggle.json

echo "
------------------------------------"
echo "Reading kaggle.json to make sure we have the token"
cat .kaggle/kaggle.json


echo "
------------------------------------"
echo "Create data directory"
mkdir -p $1
cd $1


echo "
------------------------------------"
echo "Looking where we are downloading data"
pwd

# Downloading data
kaggle competitions download -c airbus-ship-detection

unzip test_v2.zip -d test
unzip train_v2.zip -d train
unzip train_ship_segmentations_v2.csv