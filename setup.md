# Setup with Orthanc and MONAILabel

## Prerequisites
- Windows or Macos machine 
- Python 3.8 installed

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
5. Open a web browser and navigate to `http://localhost:8000/ohif` to access the OHIF app.
