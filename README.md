# SapiGo - Machine-Learning Path

## Data Engineering
Our dataset is gathered from various source such as Cattle Dataset (Kaggle) and website European Commission for the control Foot and Mooth disease (EuFMD). The dataset consist of four parts of cattle bodies (feet, gum, saliva, tongue). However, due to the limited number of images in the dataset, we have employed augmentation techniques to enhance its size and improve the accuracy of our model.

Each segment of the dataset has been uploaded by our team members on Kaggle, and the relevant links are provided below for your reference.

| Dataset | Link |
| --- | --- |
| Cattle Tongue Dataset | https://www.kaggle.com/datasets/sylviahamidah/cattle-tongue-dataset |
| Cattle Gum Dataset | https://www.kaggle.com/datasets/fauziahnisaa/cattle-gum |
| Cattle Feet Dataset | https://www.kaggle.com/datasets/fitriakd/cattle-feet-dataset |
| Cattle Saliva Dataset | https://www.kaggle.com/datasets/fitriakd/saliva-cattle-dataset |

## Machine Learning Model
We employ two distinct models. The first model we utilize is InceptionV3, which is employed in the feet, gum, and saliva models. The second model consists of a basic CNN architecture featuring three layers of Conv2D and MaxPooling, followed by flattening, two dense layers, dropout, and a final dense layer. In this CNN model, an EarlyStopping callback is also included. We create a separate model for the tongue because the performance of the CNN model is better compared to the InceptionV3 model.

## Requirement Packages/Tools
- Google Colaboratory
- Kaggle command-line tool
- zipfile 
- os 
- shutil
- random
- tensorflow
- numpy
- tensorflow
- matplotlib

## Run in Google Colaboratory
1. Download the IPYNB program files and upload them to Google Colaboratory.
2. Install the Kaggle command-line tool using the following code:
```
! pip install kaggle
```
3. Create a Kaggle API credential by creating a Kaggle account and navigating to Settings > API > Create New Token. Save the downloaded file to your Google Drive.
4. Grant permission for Google Colaboratory to access your Google Drive by running the code:
```
from google.colab import drive
drive.mount('/content/drive')
```
5. Set up the Kaggle API credentials in the Colaboratory environment with the following code:
```
! mkdir ~/.kaggle
! cp <Kaggle_API_Credential_Directory> ~/.kaggle/kaggle.json
! chmod 600 ~/.kaggle/kaggle.json
```
6. Download the dataset using the Kaggle command-line tool. For example:
```
! kaggle datasets download fitriakd/cattle-feet-dataset
```
7. Run all the program provided in the IPYNB files.
8. Save the trained model by running the code:
```
model.save("model.h5")
```
9. If the model is saved in the tmp folder, download it to your local machine for safekeeping.

## Saved Model
The trained model is saved in the .h5 format and can be accessed through the following [link](https://drive.google.com/drive/folders/1azpbBrqUfqGvPooWiTOCCMfdm6PQxIz7?usp=sharing).

