# Bash file for NLP execution

# Download GIT
git init
git config user.email "email"
git config user.name "user"
git pull https://3b93b47605174bb4a2305d0d1bc6f5a72d708130:x-oauth-basic@github.com/Ian-Lo/DATA5703_Group09 main

# May need to run this line as sudo
python3 -m pip install -r 'DevOps/requirements.txt'

# Download single file for testing code
echo 'Download GDriveDL and make executable'
curl https://raw.githubusercontent.com/matthuisman/gdrivedl/master/gdrivedl.py --output GDriveDL
chmod +x GDriveDL

mkdir Dataset_train_100k/
mkdir Dataset
./GDriveDL 'https://drive.google.com/file/d/1f_9f1oFa7qR97Q8brYsBOx033OjBr54_/view?usp=sharing'

cp dataset_01.hdf5 Dataset/train_dataset_000.hdf5
cp dataset_01.hdf5 Dataset/val_dataset_000.hdf5
