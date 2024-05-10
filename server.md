# Setting up your device as a server for MONAILabel inferences
This guide will help you set up your device as a server for MONAILabel inferences. This will allow you to run MONAILabel inferences on your device and access them from a web browser on any device connected to the same network. Or publicly using any device if you have access to your network's port managment system.

## Prerequisites
- [Setup Orthanc, MONAILabel, and OHIF](README.md) on the server device
- Admin priveleges on your device
- The device is connected to a network


## Step 1: Enable inbound port rules

### Windows System
1. Open the "Windows Defender Firewall".
2. Click on "Advanced settings".
3. Click on "Inbound Rules".
4. Click on "New Rule".
5. Select "Port" and click "Next".
6. Select "TCP" and enter "8000" in the "Specific local ports" field, if you are using the standalone OHIF server, enter the port "3000" instead of "8000".

### Macos System
1. Open the "System Preferences".
2. Click on "Security & Privacy".
3. Click on the "Firewall" tab.
4. Click on the lock icon and enter your password.
5. Click on "Firewall Options".
6. Click on the "+" icon.
7. Select "Other" and enter "8000" in the "Port Name" field, if you are using the standalone OHIF server, enter the port "3000" instead of "8000".


## Step 2: Find your device's IP address

### Windows System
1. Open the command prompt.
2. Run the following command:
    ```
    ipconfig
    ```
3. Your ip address is in the "IPv4 Address"  section.

### Macos System
1. Open the terminal.
2. Run the following command:
    ```
    ifconfig
    ```
3. Your ip address is in the "inet" section.



## Step 3: Connecting to the server

1. Open a web browser on any device connected to the same network.
2. Enter the following address in the address bar:
    ```
    http://<your_device_ip>:8000/ohif
    ```
    If you are using the standalone OHIF server, enter the following address:
    ```
    http://<your_device_ip>:3000
    ```


## Step 4: Accessing the server from outside the network (Requires port forwarding access)

We'll assume that you have setup a microsoft azure server and have access to the port forwarding system.

1. In you virtual machine portal, open the "Networking" tab and click on "Add inbound port rule".
2. Enter the following details:
    - Name: MONAILabel
    - Protocol: TCP
    - Public port: 8000 (If you are using the standalone OHIF server, enter the port "3000" instead of "8000")

3. Click on "Add" to save the rule.
4. Now you can access the server from any device using the following address:
    ```
    http://<your_public_ip>:8000/ohif
    ```
    If you are using the standalone OHIF server, enter the following address:
    ```
    http://<your_public_ip>:3000
    ```


