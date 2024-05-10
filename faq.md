# FAQ and Known Bugs
This section will help you troubleshoot common issues and answer frequently asked questions.

## How to add upload functionality to the OHIF viewer?

To add upload functionality to the OHIF viewer, you can use the OHIF uploader plugin. The plugin allows you to upload DICOM files to the Orthanc server. To add the plugin, follow these steps:

### If you are using the bundled OHIF viewer that comes with MONAILabel:
1. Navigate to the `monailabel` directory.
2. Navigate to the `MONAILabel/plugins/ohifv3/config/monai_label.js` file.
3. Add the following code to the `window.config` array:
    ```javascript

    {
         customizationService: {
            dicomUploadComponent:
                '@ohif/extension-cornerstone.customizationModule.cornerstoneDicomUploadComponent',
    },
        dicomUploadEnabled: true,
    }
    ```


### If you are using the standalone OHIF viewer:
1. Navigate to the `ohif` directory.
2. Navigate to `platform/app/public/config/docker_nginx-orthanc.json` file.
3. Add the following code to the `window.config` array:
    ```javascript

    {
         customizationService: {
            dicomUploadComponent:
                '@ohif/extension-cornerstone.customizationModule.cornerstoneDicomUploadComponent',
    },
        dicomUploadEnabled: true,
    }
    ```



## Known Bugs/Issues

### Issue 1: MONAILabel app is not downloaded properly
- **Description:** When checking the monaillabel version it is not up to date.
- **Solution:** You could be using a newer version of python when installing monailabel. Try using python 3.8 to install monailabel.
