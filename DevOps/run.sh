# Bash file for NLP execution

# Download GIT
echo 'Clone github.com/Ian-Lo/DATA5703_Group09'
git clone https://github.com/Ian-Lo/DATA5703_Group09
pip install -r 'DATA5703_Group09/DevOps/requirements.txt'


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
# Paths needs to be relative


echo 'DID YOU REMEMBER TO UNCOMMENT THE LINES YOU NEED TO EXECUTE???'
