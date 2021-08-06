This release of the x64 version of the Vaunix Lab Brick Attenuator DLL contains a test program which can
be built using Visual Studio 2017. Only the x64 configuration of the test program is supported by this SDK.

The DLL supports the LDA-102, LDA-602, LDA-302P-H, LDA-302P-1, LDA-302P-2, LDA-102E, LDA-602E, LDA-102-75,
LDA-203, LDA-602EH, LDA-602Q, LDA-906V, LDA-133, LDA-5018, LDA-5040, and LDA-906V-8 attenuators.

The DLL has ANSI-C style naming, but uses the standard x64 calling sequence. The ANSI-C style naming was chosen to simplify the task of accessing
the DLL functions from other languages such as Python or Tcl.

The DLL relies on run time libraries from the current versions of Windows, and therefore may require installation of the Visual Studio 2017 
redistributable libraries on some systems.

Note that although all attenuation values are specified in the units of .25db, or optionally .1db using the HR functions, the minimum attenuation
increment is determined by the Lab Brick hardware. The LDA-102, LDA-102E, LDA-602, LDA-602E, LDA-102-75 and LDA-302P-H attenuators have a minimum
attenuation increment of .5db. The LDA-302P-1 has a minimum attenuation increment of 1db, and the LDA-302P-2 has a minimum attenuation increment
of 2db. The Lab Brick Attenuator DLL will round down any attenuation value that has a higher resolution than the hardware supports. Thus, for example,
calling the fnLDA_SetAttenuation function with an attenuation value of 42 will result in 10.5db of attenuation on an LDA-102, and 10db of attenuation
on an LDA-302P-1.

For the high resolution attenuators, such as the LDA-602EH, LDA-602Q, and LDA-906V you should use the high resolution functions with names ending in "HR"
to take advantage of the full resolution of the devices. 

The LDA-602Q has four channels. To select a channel, use the fnLDA_SetChannel(DEVID deviceID, int channel) function. Channels are numbered from 1 to 4.
The LDA-906V-8 has eight channels, numbered from 1 to 8.

The function fnLDA_SetAttenuationHRQ(DEVID deviceID, int attenuation, int channel) combines the SetChannel and SetAttenuation functions to simplify test
software that uses the LDA-602Q quad attenuator or the LDA-906V-8 attenuator. After calling the function the channel will be set to the new channel value.

Released 5/24/2019