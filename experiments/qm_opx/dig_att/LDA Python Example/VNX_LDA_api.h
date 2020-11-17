
// --------------------------------- VNX_LDA_api.h -------------------------------------------
//
//	Include file for LabBrick attenuator API DLL (Internal version)
//	A similarly named version with only external declarations exists and is used for the SDK release
//
// (c) 2008 - 2019 by Vaunix Corporation, all rights reserved
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
//	RD	7-11-16 added support for the new HiRes attenuators, including new HiRes API entry points, and the ability
//		to run the HiRes attenuators via the existing API calls - with less resolution of course.
//
//	RD	10-31-16 added elements to our structure to handle the 4 channel attenuators.
//
//	RD	12-17-18 modified to handle our new 8 channel attenuator. Soon it will be re-coded to use
//				 a channel based model everywhere. (with foo[0] being the single channel value)
//
//	RD	8-29-19	almost done the change to a totally channel based model, added support for multi-channel operations
//				and deferred ramp and profile starts.


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
#define PROFILE_MAX 100			// Older non-hires attenuators can store 100 profile elements in their eeprom	
#define PROFILE_MAX_RAM 1000	// New FW V2.x TI based attenuators have a 1000 element RAM profile buffer
#define PROFILE_MAX_HR 50		// HiRes Attenuators only save 50 elements in their eeprom
#define MAXCHAN 64				// as of now the largest attenuator has up to 64 channels (a matrix LDA)

#define SINGLECHN 0x00
#define FOURCHN 0x01
#define EIGHTCHN 0x02
#define MULTICHN 0x03

#define PROFILE_EMPTY 0xFFFF	// we use this as our marker for an empty entry in the profile (which is an unsigned short)

// ----------- Data Types ----------------

#define DEVID unsigned int

typedef struct
{
	int	GlobalDevStatus;			// per device status
	int DevStatus[MAXCHAN];			// per channel status
	int NewChannel;					// set when we write a new channel - matches channel once the HW responds
	int NumChannels;				//  1, 4, 8, or whatever the matrix attenuator reports is the number of channels we have
	int Channel;					// the current channel 0 to 3 or 0 to 7, or 0 to 63
	int ChMask;						// the channel in bitmask form (0x01 to 0x80 for single channels in devices with up to 8 channels) or for each  bank in a matrix LDA
	int WorkingFrequency[MAXCHAN];	// the working frequency used by the HiRes units
	int MinFrequency;
	int MaxFrequency;
	int UnitScale;					// the number of .05db units in the device's HW representation (1 for HiRes, 5 for .25, 10 for .5, 20 for 1)
	int Attenuation[MAXCHAN];		// array of 8 channel values for attenuation in .05db units (used to be .25db units)
	int	MinAttenuation;
	int MaxAttenuation;
	int MinAttenStep;
	int RampStart[MAXCHAN];
	int RampStop[MAXCHAN];
	int AttenuationStep[MAXCHAN];
	int AttenuationStep2[MAXCHAN];
	int DwellTime[MAXCHAN];
	int DwellTime2[MAXCHAN];
	int IdleTime[MAXCHAN];
	int HoldTime[MAXCHAN];
	int Modebits[MAXCHAN];
	int ProfileIndex[MAXCHAN];
	int ProfileDwellTime[MAXCHAN];
	int ProfileIdleTime[MAXCHAN];
	int ProfileCount[MAXCHAN];
	int ProfileMaxLength;
	int SerialNumber;
	WCHAR ModelName[MAX_MODELNAME];
	HANDLE hDevice;
	HANDLE ghReadEvent;
	HANDLE ghReadThread;
	DWORD ReadThreadID;
	int MyDevID;
	int DevType;
	int HasMultiChan;				// 0x01 if our device is a 4 channel device, 0x02 if our device is an 8 channel device, 0x03 for the multi channel matrix devices
	int FrameCheck;
	int DevChMask;
	LARGE_INTEGER StartTime;
	LARGE_INTEGER FrameTime;		// the system time when we received the FrameNumber
	int FrameNumber;
	unsigned short Profile[MAXCHAN][PROFILE_MAX_RAM];		// this allocates 64000 ints for each of 64 devices, we try to shrink it by only allocating Uint16 elements
															// At some point we could malloc the profile space, but that may cause memory leakage
															// when applications programmers don't close our devices.
															// Now that we have matrix devices reducing this memory usage is on the T2D list

} LDAPARAMS;

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
// The first set are global to the device
#define INVALID_DEVID		0x80000000		// MSB is set if the device ID is invalid
#define DEV_CONNECTED		0x00000001		// LSB is set if a device is connected
#define DEV_OPENED			0x00000002		// set if the device is opened

// Per channel status
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
#define DEV_DEFERRED		0x00020000		// set for devices that support deferred ramp and profile commands

// Masks to select groups of status bits
#define MSK_GLSTATUS		(DEV_CONNECTED + DEV_OPENED + DEV_V2FEATURES + DEV_HIRES + DEV_DEFERRED)
#define DEVSTATUS_MASK (SWP_ACTIVE | SWP_UP	| SWP_REPEAT | SWP_BIDIRECTIONAL | PROFILE_ACTIVE)

// Feature bits for the feature DWORD 
#define DEFAULT_FEATURES	0x00000000
#define HAS_BIDIR_RAMPS		0x00000001
#define HAS_PROFILES		0x00000002
#define HAS_HIRES			0x00000004
#define HAS_4CHANNELS		0x00000008
#define HAS_8CHANNELS		0x00000010
#define HAS_LONG_PROFILE	0x00000020
#define HAS_MCHANNELS		0x00000040

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
// VNX_ATTEN_API LVSTATUS fnLDA_SetWorkingFrequencyQ(DEVID deviceID, int frequency, int channel);

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
VNX_ATTEN_API LVSTATUS fnLDA_StartProfileMC(DEVID deviceID, int mode, int chmask, bool deferred);

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