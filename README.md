# Setup with Orthanc and MONAILabel

## Prerequisites
- Windows or Macos machine 
- Python 3.9 installed

## Step 1: Install Orthanc
Official Ortnac Website for Download
https://www.orthanc-server.com/download.php

### Windows System
1. Download the Orthanc server installer from the official Orthanc website.
2. Run the installer and follow the installation instructions.
3. In the "Select Components" screen, make sure to select the "python 3.8 plugin" component.
4. Make note of the installation directory.
5. To run the Orthanc server, navigate to the installation directory and run the following command:
    ```
    Orthanc Configuration/
    ```
6. To upload Dicom Studies navigate to `http://localhost:8042/`.

### Macos System
1. Download the Orthanc server installer from the official Orthanc website.
2. Run by double clicking on the *startOrthanc.command* File in the downloaded folder.
   
   **Note:** The file runs all the plugins in the installed folder, you might get this message 'macOS Cannot Verify That This App Is Free from Malware', you need to go to Security & Privacy in the Settings and allow all of the files that are shown there while running. It will run after you allow all of them.
   
3. To upload Dicom Studies navigate to `http://localhost:8042/`.
   

## Step 2: Install MONAILabel
1. Open a command prompt or terminal.
2. Run the following command to install MONAILabel:
    ```
    pip install monailabel
    ```
3. To download a sample monai label app in your current directory run the following command:
    ```
    monailabel apps --name radiology --download --output .
    ```
4. In the same directory where you downloaded the monai app, start the MONAILabel app by running the following command. While also making sure that the orthanc server is running:
    ```
    monailabel start_server --app radiology --studies http://localhost:8042/dicom-web --conf models all
    ```
    You can read about radiology app and its models from [Monailabel Radiology App](https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/README.md) 

5. Open a web browser and navigate to `http://localhost:8000/ohif` to access the OHIF app.



# Setup OHIF with Orthanc and MONAILabel
For more flexibilty and control, you can setup[ OHIF standalone](ohif_setup.md) and connect it to Orthanc and MONAILabel.

# Server Setup
To setup a server for remote access, follow the steps in the [server setup](server.md) section.

# Add a new model
To add additional pre-trained MONAI models, follow the steps in the [server setup](adding_new_model.md) section.

# FAQ and Known Bugs
If you are facing any issues with the installation or setup, please refer to the [FAQ](faq.md) section.
# Licenses

## MONAI
MONAI is licensed under the Apache 2.0 License. See [LICENSE](https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE).

## OHIF
OHIF is licensed under the MIT License. See [LICENSE](https://github.com/OHIF/Viewers/blob/master/LICENSE).

## Orthanc
Orthan is licensed under the GNU Affero General Public License. See [LICENSE](https://github.com/jodogne/Orthanc/blob/master/LICENSE).

