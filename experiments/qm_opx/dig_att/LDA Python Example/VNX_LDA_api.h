
// --------------------------------- VNX_atten.h -------------------------------------------
//
//	Include file for LabBrick attenuator API
//
// (c) 2008 - 2014 by Vaunix Corporation, all rights reserved
//
//	RD Version 2.0	2/2014
//
//  RD 8-31-10 version for Microsoft C __cdecl calling convention
//
//	RD	This version supports the new Version 2 functions that can be used with attenuators
//		that have V2 level functionality.
//
//	RD	Merged into the 64 Bit DLL tree, version with ANSI-C style names (all x64 DLLs use a single calling convention)
//


#ifdef VNX_ATTEN64_EXPORTS
#define VNX_ATTEN_API __declspec(dllexport)
#else
#define VNX_ATTEN_API __declspec(dllimport) 
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ----------- Global Equates ------------
#define MAXDEVICES 64
#define MAX_MODELNAME 32
#define PROFILE_MAX 100

// ----------- Data Types ----------------
#define DEVID unsigned int

// ----------- Mode Bit Masks ------------
#define MODE_RFON	0x00000040			// bit is 1 for RF on, 0 if RF is off
#define MODE_INTREF	0x00000020			// bit is 1 for internal osc., 0 for external reference
#define MODE_SWEEP	0x0000001F			// bottom 5 bits are used to keep the ramp control bits				

// ----------- Profile Control -----------
#define PROFILE_ONCE	1				// play the profile once
#define PROFILE_REPEAT	2				// play the profile repeatedly
#define PROFILE_OFF		0				// stop the profile

// ----------- Command Equates -----------


// Status returns for commands
#define LVSTATUS int

#define STATUS_OK 0
#define BAD_PARAMETER			0x80010000		// out of range input -- frequency outside min/max etc.
#define BAD_HID_IO				0x80020000		// a failure in the Windows I/O subsystem
#define DEVICE_NOT_READY		0x80030000		// device isn't open, no handle, etc.
#define FEATURE_NOT_SUPPORTED	0x80040000		// the selected Lab Brick does not support this function
												// Profiles and Bi-directional ramps are only supported in
												// LDA models manufactured after

// Status returns for DevStatus

#define INVALID_DEVID		0x80000000		// MSB is set if the device ID is invalid
#define DEV_CONNECTED		0x00000001		// LSB is set if a device is connected
#define DEV_OPENED			0x00000002		// set if the device is opened
#define SWP_ACTIVE			0x00000004		// set if the device is sweeping
#define SWP_UP				0x00000008		// set if the device is ramping up
#define SWP_REPEAT			0x00000010		// set if the device is in continuous ramp mode
#define SWP_BIDIRECTIONAL	0x00000020		// set if the device is in bi-directional ramp mode
#define PROFILE_ACTIVE		0x00000040		// set if a profile is playing

// Internal values in DevStatus (changed in V2)
#define DEV_LOCKED			0x00002000		// set if we don't want read thread updates of the device parameters
#define DEV_RDTHREAD		0x00004000		// set when the read thread is running
#define DEV_V2FEATURES		0x00008000		// set for devices with V2 feature sets
#define DEV_HIRES			0x00010000		// set for HiRes devices

// Feature bits for the feature DWORD 

#define DEFAULT_FEATURES	0x00000000
#define HAS_BIDIR_RAMPS		0x00000001
#define HAS_PROFILES		0x00000002
#define HAS_HIRES			0x00000004
#define HAS_4CHANNELS		0x00000008
#define HAS_8CHANNELS		0x00000010


VNX_ATTEN_API void fnLDA_SetTraceLevel(int tracelevel, int IOtracelevel, bool verbose);		// changed 7-11-16

VNX_ATTEN_API void fnLDA_SetTestMode(bool testmode);
VNX_ATTEN_API int fnLDA_GetNumDevices();
VNX_ATTEN_API int fnLDA_GetDevInfo(DEVID *ActiveDevices);
VNX_ATTEN_API int fnLDA_GetModelNameA(DEVID deviceID, char *ModelName);
VNX_ATTEN_API int fnLDA_GetModelNameW(DEVID deviceID, wchar_t *ModelName);
VNX_ATTEN_API int fnLDA_InitDevice(DEVID deviceID);
VNX_ATTEN_API int fnLDA_CloseDevice(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetSerialNumber(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetDLLVersion();
VNX_ATTEN_API int fnLDA_GetDeviceStatus(DEVID deviceID);

VNX_ATTEN_API LVSTATUS fnLDA_SetChannel(DEVID deviceID, int channel);

VNX_ATTEN_API LVSTATUS fnLDA_SetWorkingFrequency(DEVID deviceID, int frequency);

VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuation(DEVID deviceID, int attenuation);
VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuationHR(DEVID deviceID, int attenuation);
VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuationHRQ(DEVID deviceID, int attenuation, int channel);

VNX_ATTEN_API LVSTATUS fnLDA_SetRampStart(DEVID deviceID, int rampstart);
VNX_ATTEN_API LVSTATUS fnLDA_SetRampStartHR(DEVID deviceID, int rampstart);
VNX_ATTEN_API LVSTATUS fnLDA_SetRampEnd(DEVID deviceID, int rampstop);
VNX_ATTEN_API LVSTATUS fnLDA_SetRampEndHR(DEVID deviceID, int rampstop);
VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuationStep(DEVID deviceID, int attenuationstep);
VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuationStepHR(DEVID deviceID, int attenuationstep);
VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuationStepTwo(DEVID deviceID, int attenuationstep2);
VNX_ATTEN_API LVSTATUS fnLDA_SetAttenuationStepTwoHR(DEVID deviceID, int attenuationstep2);

VNX_ATTEN_API LVSTATUS fnLDA_SetDwellTime(DEVID deviceID, int dwelltime);
VNX_ATTEN_API LVSTATUS fnLDA_SetDwellTimeTwo(DEVID deviceID, int dwelltime2);
VNX_ATTEN_API LVSTATUS fnLDA_SetIdleTime(DEVID deviceID, int idletime);
VNX_ATTEN_API LVSTATUS fnLDA_SetHoldTime(DEVID deviceID, int holdtime);

VNX_ATTEN_API LVSTATUS fnLDA_SetProfileElement(DEVID deviceID, int index, int attenuation);
VNX_ATTEN_API LVSTATUS fnLDA_SetProfileElementHR(DEVID deviceID, int index, int attenuation);
VNX_ATTEN_API LVSTATUS fnLDA_SetProfileCount(DEVID deviceID, int profilecount);
VNX_ATTEN_API LVSTATUS fnLDA_SetProfileIdleTime(DEVID deviceID, int idletime);
VNX_ATTEN_API LVSTATUS fnLDA_SetProfileDwellTime(DEVID deviceID, int dwelltime);
VNX_ATTEN_API LVSTATUS fnLDA_StartProfile(DEVID deviceID, int mode);
VNX_ATTEN_API LVSTATUS fnLDA_StartProfileMC(DEVID deviceID, int mode, int chmask, bool delayed);

VNX_ATTEN_API LVSTATUS fnLDA_SetRFOn(DEVID deviceID, bool on);

VNX_ATTEN_API LVSTATUS fnLDA_SetRampDirection(DEVID deviceID, bool up);
VNX_ATTEN_API LVSTATUS fnLDA_SetRampMode(DEVID deviceID, bool mode);
VNX_ATTEN_API LVSTATUS fnLDA_SetRampBidirectional(DEVID deviceID, bool bidir_enable);
VNX_ATTEN_API LVSTATUS fnLDA_StartRamp(DEVID deviceID, bool go);
VNX_ATTEN_API LVSTATUS fnLDA_StartRampMC(DEVID deviceID, int mode, int chmask, bool deferred);

VNX_ATTEN_API LVSTATUS fnLDA_SaveSettings(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetWorkingFrequency(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMinWorkingFrequency(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMaxWorkingFrequency(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetAttenuation(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetAttenuationHR(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetRampStart(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetRampStartHR(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetRampEnd(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetRampEndHR(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetDwellTime(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetDwellTimeTwo(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetIdleTime(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetHoldTime(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetAttenuationStep(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetAttenuationStepHR(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetAttenuationStepTwo(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetAttenuationStepTwoHR(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetRF_On(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetProfileElement(DEVID deviceID, int index);
VNX_ATTEN_API int fnLDA_GetProfileElementHR(DEVID deviceID, int index);
VNX_ATTEN_API int fnLDA_GetProfileCount(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetProfileDwellTime(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetProfileIdleTime(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetProfileIndex(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetMaxAttenuation(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMaxAttenuationHR(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMinAttenuation(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMinAttenuationHR(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMinAttenStep(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetMinAttenStepHR(DEVID deviceID);

VNX_ATTEN_API int fnLDA_GetFeatures(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetNumChannels(DEVID deviceID);
VNX_ATTEN_API int fnLDA_GetProfileMaxLength(DEVID deviceID);

#ifdef __cplusplus
	}
#endif