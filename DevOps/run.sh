# Bash file for NLP execution

# Download GIT
echo 'Clone github.com/Ian-Lo/DATA5703_Group09'
git init
git config user.email "email"
git config user.name "user"
git pull https://3b93b47605174bb4a2305d0d1bc6f5a72d708130:x-oauth-basic@github.com/Ian-Lo/DATA5703_Group09 main

pip install -r 'DevOps/requirements.txt'

# Download Dataset
echo 'Download GDriveDL and make executable'
curl https://raw.githubusercontent.com/matthuisman/gdrivedl/master/gdrivedl.py --output GDriveDL
chmod +x GDriveDL

echo 'Download datasets'
./GDriveDL 'https://googledrive.com/host/a6IU&id=1QmcBWivFeArJCVRwjKe6B6nZrTlom47W'

./GDriveDL 'https://drive.google.com/file/d/1mXjh_GBtaPz0B_asBzaWkyKrbHTKrn0t/view?usp=sharing'

./GDriveDL 'https://drive.google.com/file/d/1xJ-1mDK6DimqKH76LcKHFVPolqUOLOCD/view?usp=sharing'


# Untar files downloaded from google drive
echo 'Untaring files'
tar xvf Dataset_train_100k.tar.gz
tar xvf Dataset_test.tar.gz
tar xvf Dataset_dev.tar.gz

# Call training
# Pass path and n iters as command line parameters to training command
# Path needs to be relative
chmod +x BaseModel_pytorch/*
cd BaseModel_pytorch
python3 Test-Train.py 100000 10000 ../Dataset

# echo 'DID YOU REMEMBER TO UNCOMMENT THE LINES YOU NEED TO EXECUTE???'