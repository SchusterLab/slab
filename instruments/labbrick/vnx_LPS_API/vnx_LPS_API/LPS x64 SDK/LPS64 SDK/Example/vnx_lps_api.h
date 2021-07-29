
// --------------------------------- vnx_lps_api.h -------------------------------------------
//
//	Include file for LabBrick digital phase shifter API
//
// (c) 2008 - 2013 by Vaunix Corporation, all rights reserved
//
//	RD Version 1.0
//
//  RD 8-31-10 first version for Microsoft C __cdecl calling convention
//
//	RD	8/2013 LPS version
//
//	RD	6/2014 64 bit version


#define VNX_LPS_API __declspec(dllimport) 


#ifdef __cplusplus
extern "C" {
#endif

// ----------- Global Equates ------------
#define MAXDEVICES 64
#define MAX_MODELNAME 32

// ----------- Data Types ----------------

#define DEVID unsigned int

// ----------- Mode Bit Masks ------------

#define MODE_RFON	0x00000010			// bit is 1 for RF on, 0 if RF is off (unused)
#define MODE_INTREF	0x00000020			// bit is 1 for internal osc., 0 for external reference (unused)
#define MODE_SWEEP	0x00000017			// bottom 5 bits are used to keep the sweep control bits
										// bit 3 is the RF_ON bit, which is ignored, and hence zeroed out

// Bit masks and equates for the Ramp command byte (stored in Sweep_mode, and reported also in Status)
#define LPS_SWP_BIDIR			0x10	// MASK: bit = 0 for ramp style sweep, 1 for triangle style sweep
#define LPS_SWP_DIRECTION		0x04	// MASK: bit = 0 for sweep up, 1 for sweep down 
#define LPS_SWP_CONTINUOUS		0x02	// MASK: bit = 1 for continuous sweeping
#define LPS_SWP_ONCE			0x01	// MASK: bit = 1 for single sweep



// ----------- Command Equates -----------


// Status returns for commands
#define LPSTATUS int

#define STATUS_OK 0
#define BAD_PARAMETER 0x80010000		// out of range input -- frequency outside min/max etc.
#define BAD_HID_IO    0x80020000
#define DEVICE_NOT_READY 0x80030000		// device isn't open, no handle, etc.

// Status returns for DevStatus
	
#define INVALID_DEVID		0x80000000		// MSB is set if the device ID is invalid
#define DEV_CONNECTED		0x00000001		// LSB is set if a device is connected
#define DEV_OPENED			0x00000002		// set if the device is opened
#define SWP_ACTIVE			0x00000004		// set if the device is sweeping
#define SWP_UP				0x00000008		// set if the device is ramping up
#define SWP_REPEAT			0x00000010		// set if the device is in continuous ramp mode
#define SWP_BIDIRECTIONAL	0x00000020		// set if the device is in bi-directional ramp mode
#define PROFILE_ACTIVE		0x00000040		// set if a profile is playing

// Internal values in DevStatus -- read only!
#define DEV_LOCKED	  0x00002000		// set if we don't want read thread updates of the device parameters
#define DEV_RDTHREAD  0x00004000		// set when the read thread is running


VNX_LPS_API void fnLPS_SetTestMode(bool testmode);
VNX_LPS_API int fnLPS_GetNumDevices();
VNX_LPS_API int fnLPS_GetDevInfo(DEVID *ActiveDevices);
VNX_LPS_API int fnLPS_GetModelNameA(DEVID deviceID, char *ModelName);
VNX_LPS_API int fnLMS_GetModelNameW(DEVID deviceID, wchar_t *ModelName);
VNX_LPS_API int fnLPS_InitDevice(DEVID deviceID);
VNX_LPS_API int fnLPS_CloseDevice(DEVID deviceID);
VNX_LPS_API int fnLPS_GetSerialNumber(DEVID deviceID);
VNX_LPS_API int fnLPS_GetDeviceStatus(DEVID deviceID);
VNX_LPS_API int fnLPS_GetDLLVersion();


VNX_LPS_API LPSTATUS fnLPS_SetPhaseAngle(DEVID deviceID, int phase);
VNX_LPS_API LPSTATUS fnLPS_SetWorkingFrequency(DEVID deviceID, int frequency);
VNX_LPS_API LPSTATUS fnLPS_SetRampStart(DEVID deviceID, int rampstart);
VNX_LPS_API LPSTATUS fnLPS_SetRampEnd(DEVID deviceID, int rampstop);
VNX_LPS_API LPSTATUS fnLPS_SetPhaseAngleStep(DEVID deviceID, int phasestep);
VNX_LPS_API LPSTATUS fnLPS_SetPhaseAngleStepTwo(DEVID deviceID, int phasestep2);
VNX_LPS_API LPSTATUS fnLPS_SetDwellTime(DEVID deviceID, int dwelltime);
VNX_LPS_API LPSTATUS fnLPS_SetDwellTimeTwo(DEVID deviceID, int dwelltime2);
VNX_LPS_API LPSTATUS fnLPS_SetIdleTime(DEVID deviceID, int idletime);
VNX_LPS_API LPSTATUS fnLPS_SetHoldTime(DEVID deviceID, int holdtime);

VNX_LPS_API LPSTATUS fnLPS_SetProfileElement(DEVID deviceID, int index, int phaseangle);
VNX_LPS_API LPSTATUS fnLPS_SetProfileCount(DEVID deviceID, int profilecount);
VNX_LPS_API LPSTATUS fnLPS_SetProfileIdleTime(DEVID deviceID, int idletime);
VNX_LPS_API LPSTATUS fnLPS_SetProfileDwellTime(DEVID deviceID, int dwelltime);
VNX_LPS_API LPSTATUS fnLPS_StartProfile(DEVID deviceID, int mode);

VNX_LPS_API LPSTATUS fnLPS_SetRampDirection(DEVID deviceID, bool up);
VNX_LPS_API LPSTATUS fnLPS_SetRampMode(DEVID deviceID, bool mode);
VNX_LPS_API LPSTATUS fnLPS_SetRampBidirectional(DEVID deviceID, bool bidir_enable);
VNX_LPS_API LPSTATUS fnLPS_StartRamp(DEVID deviceID, bool go);

VNX_LPS_API LPSTATUS fnLPS_SaveSettings(DEVID deviceID);

VNX_LPS_API int fnLPS_GetPhaseAngle(DEVID deviceID);
VNX_LPS_API int fnLPS_GetWorkingFrequency(DEVID deviceID);

VNX_LPS_API int fnLPS_GetRampStart(DEVID deviceID);
VNX_LPS_API int fnLPS_GetRampEnd(DEVID deviceID);
VNX_LPS_API int fnLPS_GetDwellTime(DEVID deviceID);
VNX_LPS_API int fnLPS_GetDwellTimeTwo(DEVID deviceID);
VNX_LPS_API int fnLPS_GetIdleTime(DEVID deviceID);
VNX_LPS_API int fnLPS_GetHoldTime(DEVID deviceID);
VNX_LPS_API int fnLPS_GetPhaseAngleStep(DEVID deviceID);
VNX_LPS_API int fnLPS_GetPhaseAngleStepTwo(DEVID deviceID);

VNX_LPS_API int fnLPS_GetProfileElement(DEVID deviceID, int index);
VNX_LPS_API int fnLPS_GetProfileCount(DEVID deviceID);
VNX_LPS_API int fnLPS_GetProfileDwellTime(DEVID deviceID);
VNX_LPS_API int fnLPS_GetProfileIdleTime(DEVID deviceID);
VNX_LPS_API int fnLPS_GetProfileIndex(DEVID deviceID);

VNX_LPS_API int fnLPS_GetMaxPhaseShift(DEVID deviceID);
VNX_LPS_API int fnLPS_GetMinPhaseShift(DEVID deviceID);
VNX_LPS_API int fnLPS_GetMinPhaseStep(DEVID deviceID);

VNX_LPS_API int fnLPS_GetMaxWorkingFrequency(DEVID deviceID);
VNX_LPS_API int fnLPS_GetMinWorkingFrequency(DEVID deviceID);

#ifdef __cplusplus
}
#endif