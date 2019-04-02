# Upload kaggle.json before launching the script
# Modify directory if necessary

# data_dir example: ../data/airbus_ship_detection
# Token kaggle dir example: ~/kaggle.json

sudo pip install kaggle --upgrade

# Create data directory
mkdir -p $1
cd $1
mkdir .kaggle
ls -a

# Copying kaggle token 
cp $2 .kaggle/
ls .kaggle/

chmod 600 .kaggle/kaggle.json
cat .kaggle/kaggle.json

# Downloading data
kaggle competitions download -c airbus-ship-detection

unzip test_v2.zip -d test
unzip train_v2.zip -d train
unzip train_ship_segmentations_v2.csv