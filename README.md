# Mental_health_portal_ML_architecture
This repo include the ML pipeline to train the machine to predict whther individual isdepressed or not.

# Dataset
Gain access to the DAIC-WOZ database and download the zip files to your project directory by running the following command in your shell:
> wget -r -np -nH --cut-dirs=3 -R index.html --user=daicwozuser --ask-password  http://dcapswoz.ict.usc.edu/wwwdaicwoz/

# Data Extraction
1. Run extract_from_zip.py to extract the wav files of the interviews and interview transcription csv files from the zip files.
2. Run segmentation.py to create segmented wav files for each participant (silence and the virtual interviewer's speech removed). Feature extraction is performed on the segmented wav files.

Afterwards, follow commands in Mental_Health_Portal_ML_Architecture.ipynb file for feature extraction and training model.
