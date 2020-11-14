from ctypes import *
vnx=cdll.VNX_atten64
vnx.fnLDA_SetTestMode(False)
DeviceIDArray = c_int * 20
Devices = DeviceIDArray()

# GetNumDevices will determine how many LDA devices are availible
numDevices = vnx.fnLDA_GetNumDevices()
print(str(numDevices), ' device(s) found')

# GetDevInfo generates a list, stored in the devices array, of
# every availible LDA device attached to the system
# GetDevInfo will return the number of device handles in the array
dev_info = vnx.fnLDA_GetDevInfo(Devices)
print('GetDevInfo returned', str(dev_info))

# GetSerialNumber will return the devices serial number
ser_num = vnx.fnLDA_GetSerialNumber(Devices[0])
print('Serial number:', str(ser_num))

#InitDevice wil prepare the device for operation
init_dev = vnx.fnLDA_InitDevice(Devices[0])
print('InitDevice returned', str(init_dev))

print()

#GetNumChannels will return the number of channels on the device
num_channels = vnx.fnLDA_GetNumChannels(Devices[0]);
if num_channels < 1:
    num_channels = 1

# Input desired attenuation level for channel 1
print('Set attenuation level for channel 1', end = '')
atten = float(input(': '))
attenuation = atten / .05
atten = round(attenuation)

# Select channel 1
channel_1 = vnx.fnLDA_SetChannel(Devices[0], 1)
if channel_1 != 0:
    print('SetChannel returned error', channel_1)

# Set attenuation level for channe 1
result_1 = vnx.fnLDA_SetAttenuationHR(Devices[0], int(atten))
if result_1 != 0:
    print('SetAttenuationHR returned error', result_1)

# Get channel 1 attenuation
result_1 = vnx.fnLDA_GetAttenuationHR(Devices[0])

# Use a for loop to loop through the remainder of the channels
for i in range(2, num_channels + 1):

    # Select the channel
    channel = vnx.fnLDA_SetChannel(Devices[0], i)
    if channel != 0:
        print('SetChannel returned an error', str(channel))

    #Input desired attenuation for channel i
    print('Set attenuation level for channel', i, end = '')
    atten = float(input(': '))
    attenuation = atten / .05
    atten = round(attenuation)
    
    #Set the attenuation for channel i
    result = vnx.fnLDA_SetAttenuationHR(Devices[0], int(atten))
    if result != 0:
        print('SetAttenuationHR returned error', result)

print()

# Display attenuation level for channel 1
if result_1 < 0:
    print('GetAttenuationHR returned error', result)

else:
    atten_db = result_1 / 20
    print('Channel 1 Attenuation:', atten_db)

# Use a for loop to display the attenuation levels for any additional channels
for i in range (2, num_channels +1):
    if i > 0:

        # Select channel i
        channel = vnx.fnLDA_SetChannel(Devices[0], i)

        # Get channel i's attenuation
        result = vnx.fnLDA_GetAttenuationHR(Devices[0])

        # Display channel i's attenuation
        if result < 0:
            print('GetAttenuationHR returned error', result)

        else:
            atten_db = result / 20
            print('Channel', i, 'Attenuation:', atten_db)

# Always close the device when done with it
result = vnx.fnLDA_CloseDevice(Devices[0])

if result != 0:
    print('CloseDevice returned an error', result)
