// LDA64Test.cpp : Defines the entry point for the console application.
//
//	This is an example of how to use the Vaunix Lab Brick attenuator 64 bit DLL from a Windows application
//
//	Modified 7/2016 to show an example of using the new HiRes functions
//
//	Modified 10/31/2016 to show an example of using the Quad Attenuator functions
//  Modified 1/6/2018 to support simple multi-channel attenuator operations

#include "stdafx.h"
#include "vnx_LDA_api.h"

// ------------------------------ - Equates-----------------------------------------------
#define CL_SWP_DIRECTION		0x04	// MASK: bit = 0 for ramp up, 1 for ramp down 
#define CL_SWP_CONTINUOUS		0x02	// MASK: bit = 1 for continuous ramping
#define CL_SWP_ONCE				0x01	// MASK: bit = 1 for single ramp
#define CL_SWP_BIDIRECTIONALLY	0x10	// MASK: bit = 1 for bidirectional ramps (V2 LDA's only)


// ------------------------------- Allocations -------------------------------------------

static DEVID MyDevices[MAXDEVICES];				// I have statically allocated this array for convenience
												// It holds a list of device IDs for the connected devices
												// They are stored starting at MyDevices[0]

static char MyDeviceNameA[MAX_MODELNAME];		// NB -- this is a single byte char array for testing the ASCII name function
static wchar_t MyDeviceNameW[MAX_MODELNAME];	// NB -- this is a WCHAR array for testing the Unicode name function

static wchar_t errmsg[32];						// For the status->string converter
static char cModelName[32];						// buffer for the model name

static string sDevName = "ATN-001";				// device name string
static bool gbWantOneDevice = FALSE;

static int DevNum = 0;				// the device we should work with.
static int DevRange = 1;			// the number of devices we'll send the command to
static int NumDevices = 0;			// used to store the actual number of devices found


// --------------------------- Variables -------------------------------------------------

static int IdleTime = 1;			// default idle time is 1 ms
static int HoldTime = 1;			// default hold time is 1 ms
static int AStart = 0;				// default atten start level is 0 db.
static int AStop = 252;				// default atten stop, for most devices this is 63 db so we use that.
static int Dwell = 1000;			// default dwell time is 1 second for first ramp phase
static int Dwell2 = 1000;			// default dwell time is 1 second for second ramp phase (V2 LDA's only)
static int AStep = 2;				// default step size is .5db, some LDA's have larger minimum steps
static int AStep2 = 2;				// default second phase step size for LDA's that support bidirectional ramps

static int WorkingFrequency = 0;	// working frequency for the HiRes attenuators
static float Attenuation = 0;		// default attenuation is 0db, entered as a floating point value

static int ScaledAttenuation = 0;	// temporary storage for scaled attenuation values
static int SerialNumber = 0;		// used to hold the serial number for the get serial number command

static int RFOnOff = 1;				// really used as a bool -- if non zero, turn on the RF output

static int Sweep_mode = 0;			// used to control the sweep mode
static int GetParam = 0;			// the low byte is the GET command byte

static int ProfileIndex = 0;		// the element in the profile we want to set
static int ProfileLength = 0;		// the length of the profile
static int ProfileValue = 0;		// the profile element's value

static int Channel = 1;				// we just default to the first channel

bool gbWantSetIdle = FALSE;
bool gbWantSetHold = FALSE;
bool gbWantSetAStart = FALSE;
bool gbWantSetAStop = FALSE;
bool gbWantSetDwell = FALSE;
bool gbWantSetDwell2 = FALSE;
bool gbWantStartSweep = FALSE;
bool gbWantSetAStep = FALSE;
bool gbWantSetAStep2 = FALSE;
bool gbWantSetWorkingFrequency = FALSE;
bool gbWantSetAttenuation = FALSE;
bool gbWantSaveSettings = FALSE;
bool gbWantGetParam = FALSE;
bool gbBatchMode = FALSE;
bool gbWantSetRFOnOff = FALSE;
bool gbQuietMode = FALSE;
bool gbWantSetProfileElement = FALSE;
bool gbWantSetProfileLength = FALSE;
bool gbWantChannel = FALSE;




// ------------------------------- Support Routines --------------------------------------

void PrintHelp()
{
	printf("Vaunix Attenuator Demonstration\n");
	printf("\n");
	printf("Hit CTRL+C to exit\n");
	printf("\n");

	printf(" --- Overall modes and device selection. Defaults to first device ---\n");
	printf("  -d i n 	Select the devices to work with, i is the device number (1,2,3, etc.)\n");
	printf("     		and n is the number of devices to apply the command to.\n");
	printf("     		-d 1 2 applies the commands to attenuators 1 and 2.\n");
	printf("     		-d 2 3 applies the commands to attenuators 2, 3 and 4.\n");
	printf("  -y		Save the current settings in the device.\n");
	printf("\n");
	printf("  -b		Batch mode, exit immediately after sending commands to the Lab Bricks.\n");
	printf("  -q		Quiet mode, skip most outputs.\n");
	printf("\n");

	printf(" --- Commands to set parameters and start ramp --- \n");
	printf("  -c n      Set the active channel\n");
	printf("  -f nn     Set working frequency, nn is working frequency in MHz\n");
	printf("  -a nn     Set attenuation, nn is attenuation in db units\n");
	printf("  -w nn     Set idle time between attenuator ramps, nn is time in ms.\n");
	printf("  -h nn     Set hold time between ramp phases\n");
	printf("  -s nn     Set ramp start value, nn is start value in .25db units\n");
	printf("  -e nn     Set ramp end value, nn is end value in .25db units, p is ramp phase\n");
	printf("  -t p nn   Set time to dwell on each attenuation value, nn is time in ms., p is ramp phase 1 or 2\n");

	printf("  -i p nn   Set attenuation ramp increment, nn is the increment\n");
	printf("            in .25 db units. p is ramp phase 1 or 2\n");
	printf("  -g n      Start a ramp, 1 = once upwards, 2 = continuous upwards\n");
	printf("            5 = once down, 6 = continuous down, 17 = bidirectional once,\n");
	printf("            18 = continuous bidirectional ramps, 0 to stop\n");

	printf("\n");


}

// -------------------- - MakeLower------------------------------

wchar_t MakeLowerW(wchar_t &wc)
{
	return wc = towlower(wc);
}

// --------------------------------------------------------------

#define MAX_MSG 32

/* A function to display an error status as a Unicode string */
wchar_t* fnLDA_perror(LVSTATUS status) {
	wcscpy_s(errmsg, MAX_MSG, L"STATUS_OK");
	if (BAD_PARAMETER == status) wcscpy_s(errmsg, MAX_MSG, L"BAD_PARAMETER");
	if (BAD_HID_IO == status) wcscpy_s(errmsg, MAX_MSG, L"BAD_HID_IO");
	if (DEVICE_NOT_READY == status) wcscpy_s(errmsg, MAX_MSG, L"DEVICE_NOT_READY");
	if (FEATURE_NOT_SUPPORTED == status) wcscpy_s(errmsg, MAX_MSG, L"FEATURE_NOT_SUPPORTED");
	if (INVALID_DEVID == status) wcscpy_s(errmsg, MAX_MSG, L"INVALID_DEVID");

	return errmsg;
}

// -- one way to check for errors --
void CheckAPISet(LVSTATUS status)
{
	if (status & 0x80000000)
	{
		wprintf(L"*** Error: LDA API returned status = %x, %s ***\n", status, fnLDA_perror(status));
	}

}

/* A function to display the status as a Unicode string */
wchar_t* fnLDA_pstatus(LVSTATUS status) {
	wcscpy_s(errmsg, MAX_MSG, L"STATUS_OK");

	// Status returns for DevStatus
	if (INVALID_DEVID == status) wcscpy_s(errmsg, MAX_MSG, L"INVALID_DEVID");
	if (DEV_CONNECTED == status) wcscpy_s(errmsg, MAX_MSG, L"DEV_CONNECTED");
	if (DEV_OPENED == status) wcscpy_s(errmsg, MAX_MSG, L"DEV_OPENED");
	if (SWP_ACTIVE == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_ACTIVE");
	if (SWP_UP == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_UP");
	if (SWP_REPEAT == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_REPEAT");
	if (SWP_BIDIRECTIONAL == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_BIDIRECTIONAL");
	if (PROFILE_ACTIVE == status) wcscpy_s(errmsg, MAX_MSG, L"PROFILE_ACTIVE");

	return errmsg;
}




// ParseCommandLine() will return FALSE to indicate that we received an invalid
// command or should abort for another reason.
bool ParseCommandLine(int argc, _TCHAR *argv[])
{
	int RampPhase;

	enum {
		wantDash, wantDevSubstring, wantIdle, wantAStart, wantAStop, wantDwell, wantAStep,
		wantAtten, wantSetRFOnOff, wantSweep, wantGetParam, wantDevID, wantDevRange,
		wantDwell2, wantAStep2, wantHold, wantDwellPhase, wantStepPhase, wantWorkingFrequency,
		wantChannel
	} state = wantDash;

	for (int i = 1; i < argc; ++i) {
		// Convert each argument to lowercase
		wstring thisParam(argv[i]);
		for_each(thisParam.begin(), thisParam.end(), MakeLowerW);

		if (state == wantDash)
		{
			if ('-' != thisParam[0])
			{
				printf("\n *** Error in command line syntax *** \n");
				PrintHelp();
				return FALSE;
			}
			// remove the dash from the front of the string
			thisParam = wstring(thisParam.begin() + 1, thisParam.end());

			// Identify the command line arguments
			if (L"d" == thisParam) {
				state = wantDevID;
			}
			else if (L"w" == thisParam) {
				gbWantSetIdle = TRUE;
				state = wantIdle;
			}
			else if (L"s" == thisParam) {
				gbWantSetAStart = TRUE;
				state = wantAStart;
			}
			else if (L"e" == thisParam) {
				gbWantSetAStop = TRUE;
				state = wantAStop;
			}
			else if (L"t" == thisParam) {
				state = wantDwellPhase;
			}
			else if (L"i" == thisParam) {
				state = wantStepPhase;
			}
			else if (L"a" == thisParam) {
				gbWantSetAttenuation = TRUE;
				state = wantAtten;
			}
			else if (L"g" == thisParam) {
				gbWantStartSweep = TRUE;
				state = wantSweep;
			}
			else if (L"r" == thisParam) {
				gbWantSetRFOnOff = TRUE;
				state = wantSetRFOnOff;
			}
			else if (L"y" == thisParam) {
				gbWantSaveSettings = TRUE;
				state = wantDash;
			}
			else if (L"b" == thisParam) {
				gbBatchMode = TRUE;
				state = wantDash;
			}
			else if (L"q" == thisParam) {
				gbQuietMode = TRUE;
				state = wantDash;
			}
			else if (L"h" == thisParam) {
				gbWantSetHold = TRUE;
				state = wantHold;
			}
			else if (L"f" == thisParam) {
				gbWantSetWorkingFrequency = TRUE;
				state = wantWorkingFrequency;
			}
			else if (L"c" == thisParam) {
				gbWantChannel = TRUE;
				state = wantChannel;
			}

			else {
				// this case is for "-?" and any argument we don't recognize
				PrintHelp();
				return FALSE;	// don't continue
			}
		}

		else {

			// save the whole substring and do conversions for each argument type

			switch (state){

			case wantDwellPhase:
				RampPhase = _wtoi(thisParam.c_str());
				if (RampPhase == 1){
					gbWantSetDwell = TRUE;
					state = wantDwell;
				}
				else if (RampPhase == 2){
					gbWantSetDwell2 = TRUE;
					state = wantDwell2;
				}
				else state = wantDash;		// phase value is wrong, not much we can do about it...
				break;

			case wantStepPhase:
				RampPhase = _wtoi(thisParam.c_str());
				if (RampPhase == 1){
					gbWantSetAStep = TRUE;
					state = wantAStep;
				}
				else if (RampPhase == 2){
					gbWantSetAStep2 = TRUE;
					state = wantAStep2;
				}
				else state = wantDash;		// phase value is wrong, not much we can do about it...
				break;

			case wantIdle:
				IdleTime = _wtoi(thisParam.c_str());		// convert to a int
				state = wantDash;
				break;

			case wantHold:
				HoldTime = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantDevID:
				DevNum = _wtoi(thisParam.c_str());
				state = wantDevRange;
				break;

			case wantChannel:
				Channel = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantDevRange:
				DevRange = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantAStart:
				AStart = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantAStop:
				AStop = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantDwell:
				Dwell = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantDwell2:
				Dwell2 = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantAStep:
				AStep = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantAStep2:
				AStep2 = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantAtten:
				Attenuation = (float)_wtof(thisParam.c_str());	// cast to a float, _wtof actually returns a double
				state = wantDash;
				break;

			case wantWorkingFrequency:
				WorkingFrequency = (float)_wtof(thisParam.c_str());	// cast to a float, _wtof actually returns a double
				state = wantDash;
				break;


			case wantSetRFOnOff:
				RFOnOff = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantSweep:
				Sweep_mode = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			}
		}
	}

	if (state != wantDash) {
		// we are expecting an argument, if we didn't get one then print the help message
		PrintHelp();
		return FALSE;
	}

	// It's OK to continue
	return TRUE;
}
// ------------- Command Line Main ---------------------------------------------------

int _tmain(int argc, _TCHAR* argv[])
{
	int i, j, k;
	int iDev;
	int itemp;
	bool bTemp;
	float ftemp;
	int NumChannels;		// temporary storage for the number of channels in a device

	if (!ParseCommandLine(argc, argv))
		return 0;

	if (!gbQuietMode) printf("Lab Brick Multi Attenuator Demonstration Program\n");

	//	 -- convert the user's device number to our internal MyDevices array index and check our device range --

	DevNum = DevNum - 1;

	if (DevNum < 0) DevNum = 0;
	if (DevRange < 1) DevRange = 1;
	if (DevRange > MAXDEVICES) DevRange = MAXDEVICES;

	if (DevNum  > MAXDEVICES - 1) DevNum = MAXDEVICES - 1;
	if ((DevNum + DevRange) > MAXDEVICES) DevRange = MAXDEVICES - DevNum;

	// at this point our device starting index and number of devices should be reasonable...
	if (!gbQuietMode) printf("Starting device number = %d, using %d device[s]\n", DevNum + 1, DevRange);

	// --- if TestMode = TRUE then the dll will fake the hardware ---
	fnLDA_SetTestMode(FALSE);


	// --- Use the tracing control function to turn on debug messages
	fnLDA_SetTraceLevel(0, 0, false);
	//fnLDA_SetTraceLevel(3, 3, true);

	i = fnLDA_GetNumDevices();

	if (i == 0){
		printf("No device found\n");
	}

	if (i == 1){
		if (!gbQuietMode) printf("Found %d Device\n", i);

	}
	else {
		if (!gbQuietMode) printf("Found %d Devices\n", i);
	}

	// -- warn the user if he or she expects more devices than we have --
	if (DevRange > i){
		printf(" Warning - not enough attenuators are connected\n");
	}

	NumDevices = fnLDA_GetDevInfo(MyDevices);

	if (!gbQuietMode) printf("Got Device Info for %d Device[s]\n", NumDevices);


	if (NumDevices > 0)	// do we have a device?
	{
		for (j = 0; j < NumDevices; j++){

			// --- print out the first device's name ---
			if (!gbQuietMode){
				itemp = fnLDA_GetModelNameA(MyDevices[j], MyDeviceNameA);
				printf("Device %d is an %s \n", j + 1, MyDeviceNameA);
			}

			// --- print out the device's serial number ---
			if (!gbQuietMode){
				itemp = fnLDA_GetSerialNumber(MyDevices[j]);
				if (itemp >= 0)
					printf("Device %d has serial number %d \n", j + 1, itemp);
			}


			// --- We need to init the device (open it) before we can do anything else ---
			itemp = fnLDA_InitDevice(MyDevices[j]);

			if (itemp){
				printf("InitDevice returned error code %x\n", itemp);
			}

			// --- Lets see if we got the device's parameters ---
			if (!gbQuietMode) {

				// Display the number of channels for this device
				NumChannels = fnLDA_GetNumChannels(MyDevices[j]);
				if (NumChannels >= 0){
					if (NumChannels == 1) printf("Single Channel Device\n");
					else printf("Device has %d Channels\n", NumChannels);
				}
				else
					CheckAPISet(NumChannels);

				// show device wide parameters
				printf("Attenuation Range:\n");
				itemp = fnLDA_GetMinAttenuation(MyDevices[j]);
				ftemp = itemp * .25;
				if (itemp >= 0)
					printf("Minimum Attenuation = %.2f db\n", ftemp);
				else
					CheckAPISet(itemp);

				itemp = fnLDA_GetMaxAttenuation(MyDevices[j]);
				ftemp = itemp * .25;
				if (itemp >= 0)
					printf("Maximum Attenuation = %.2f db\n", ftemp);
				else
					CheckAPISet(itemp);

				if (fnLDA_GetFeatures(MyDevices[j]) & HAS_HIRES)
				{
					printf("Working Frequency Range:\n");
					itemp = fnLDA_GetMinWorkingFrequency(MyDevices[j]);
					ftemp = itemp / 10;		// frequency is in 100KHz units
					if (itemp >= 0)
						printf("Minimum Frequency = %.2f Mhz\n", ftemp);
					else
						CheckAPISet(itemp);

					itemp = fnLDA_GetMaxWorkingFrequency(MyDevices[j]);
					ftemp = itemp / 10;		// frequency is in 100KHz units
					if (itemp >= 0)
						printf("Maximum Frequency = %.2f Mhz\n", ftemp);
					else
						CheckAPISet(itemp);
				}

				// if we have more than one channel, show the parameters for each channel
				for (i = 0; i < NumChannels; i++)
				{
					if (NumChannels > 1)
					{
						printf("\nParameters for device %d channel %d\n", j + 1, i + 1);
						fnLDA_SetChannel(MyDevices[j], i + 1);	// the channel argument runs from 1 to N channels
						CheckAPISet(itemp);
					}

					if (fnLDA_GetFeatures(MyDevices[j]) & HAS_HIRES)
					{
						// Display the frequency related parameters
						itemp = fnLDA_GetWorkingFrequency(MyDevices[j]);
						ftemp = itemp / 10;	// working frequency is in 100KHz units
						if (itemp >= 0)
							printf("Working Frequency = %.2f Mhz\n", ftemp);
						else
							CheckAPISet(itemp);
					}

					itemp = fnLDA_GetAttenuationHR(MyDevices[j]);
					ftemp = (float)itemp / 20;
					if (itemp >= 0)
						printf("Attenuation = %.2f db\n", ftemp);
					else
						CheckAPISet(itemp);

					itemp = fnLDA_GetRampStart(MyDevices[j]);
					ftemp = itemp * .25;
					if (itemp >= 0)
						printf("Ramp Start Level = %.2f db\n", ftemp);
					else
						CheckAPISet(itemp);

					itemp = fnLDA_GetRampEnd(MyDevices[j]);
					ftemp = itemp * .25;
					if (itemp >= 0)
						printf("Ramp End Level = %.2f db\n", ftemp);
					else
						CheckAPISet(itemp);

					itemp = fnLDA_GetAttenuationStep(MyDevices[j]);
					ftemp = itemp * .25;
					if (itemp >= 0)
						printf("First Phase Ramp Attenuation Step Size = %.2f db\n", ftemp);
					else
						CheckAPISet(itemp);

					if (fnLDA_GetFeatures(MyDevices[j]) > 0)
					{
						itemp = fnLDA_GetAttenuationStepTwo(MyDevices[j]);
						ftemp = itemp * .25;
						if (itemp >= 0)
							printf("Second Phase Ramp Attenuation Step Size = %.2f db\n", ftemp);
						else
							CheckAPISet(itemp);
					}

					itemp = fnLDA_GetDwellTime(MyDevices[j]);
					if (itemp >= 0)
						printf("First Phase Ramp Dwell Time = %d\n", itemp);
					else
						CheckAPISet(itemp);

					if (fnLDA_GetFeatures(MyDevices[j]) > 0)
					{
						itemp = fnLDA_GetDwellTimeTwo(MyDevices[j]);
						if (itemp >= 0)
							printf("Second Phase Ramp Dwell Time = %d\n", itemp);
						else
							CheckAPISet(itemp);
					}

					itemp = fnLDA_GetIdleTime(MyDevices[j]);
					if (itemp >= 0)
						printf("Ramp Idle Time = %d\n", itemp);
					else
						CheckAPISet(itemp);

					if (fnLDA_GetFeatures(MyDevices[j]) > 0)
					{
						itemp = fnLDA_GetHoldTime(MyDevices[j]);
						if (itemp >= 0)
							printf("Ramp Hold Time = %d\n", itemp);
						else
							CheckAPISet(itemp);
					}

					// --- Show if the RF output is on ---
					itemp = fnLDA_GetRF_On(MyDevices[j]);
					if (itemp >= 0)
					{
						if (itemp != 0)
						{
							printf("RF ON\n");
						}
						else
						{
							printf("RF OFF\n");
						}
					}
					else
					{
						CheckAPISet(itemp);
					}

					printf("\n");

				} // end of our loop over channels

				

			} // end of our quiet mode case

		} // end of the for loop over the devices

		// if the user is trying to control a device we don't have, then quit now

		if (DevNum > NumDevices - 1){
			for (j = 0; j < NumDevices; j++)
			{
				itemp = fnLDA_CloseDevice(MyDevices[j]);
			}
			printf("First selected device is not attached, exiting.\n");
			return 0;			// quit - nothing else to do
		}

		// if the user is trying to control more devices than we have, reduce the number of devices in the group
		if ((DevNum + DevRange) > NumDevices){
			DevRange = NumDevices - DevNum;
			printf("Not enough attenuators connected, using %d devices.\n", DevRange);
		}

		// ------------- Now we'll set the requested device or devices with new parameters -------------

		if (!gbQuietMode)printf("Setting the attenuator parameters..\n");

		for (iDev = DevNum; iDev < DevNum + DevRange; iDev++)
		{

			if (gbWantChannel)
			{
				if (!gbQuietMode) printf("Setting the channel for device %d to %d", iDev + 1, Channel);
				itemp = fnLDA_SetChannel(MyDevices[iDev], Channel);
			}


			// --- Lets set the attenuation for the channel first ---
			if (gbWantSetAttenuation)
			{

				// using the HiRes API function with .05db units
				ScaledAttenuation = (int)(Attenuation * 20);

				// Set the selected channel with the attenuation
				if (!gbQuietMode) printf("Setting the attenuation for Channel %d to %.2f db\n", Channel, ((float)(ScaledAttenuation)/20));
				itemp = fnLDA_SetAttenuationHRQ(MyDevices[iDev], ScaledAttenuation, Channel);

			}

			// --- and then do whatever else the user requested ---

			if (gbWantSetDwell)
			{
				if (!gbQuietMode) printf("Setting the first phase dwell time for device %d to %d \n", iDev + 1, Dwell);
				itemp = fnLDA_SetDwellTime(MyDevices[iDev], Dwell);
				CheckAPISet(itemp);
			}

			if (gbWantSetAStart)
			{
				ftemp = (float)AStart / 4;
				if (!gbQuietMode) printf("Setting the ramp start for device %d to %.2f db \n", iDev + 1, ftemp);
				itemp = fnLDA_SetRampStart(MyDevices[iDev], AStart);
				CheckAPISet(itemp);
			}

			if (gbWantSetAStop)
			{
				ftemp = (float)AStop / 4;
				if (!gbQuietMode) printf("Setting ramp end for device %d to %.2f db \n", iDev + 1, ftemp);
				itemp = fnLDA_SetRampEnd(MyDevices[iDev], AStop);
				CheckAPISet(itemp);
			}

			if (gbWantSetAStep)
			{
				ftemp = (float)AStep / 4;
				if (!gbQuietMode) printf("Setting the first phase attenuation step for device %d to %.2f db \n", iDev + 1, ftemp);
				itemp = fnLDA_SetAttenuationStep(MyDevices[iDev], AStep);
				CheckAPISet(itemp);
			}

			if (gbWantSetIdle)
			{
				if (!gbQuietMode) printf("Setting the idle time between ramps for device %d to %d ms. \n", iDev + 1, IdleTime);
				itemp = fnLDA_SetIdleTime(MyDevices[iDev], IdleTime);
				CheckAPISet(itemp);
			}

			if (gbWantSetRFOnOff)
			{
				if (RFOnOff == 0)
				{
					bTemp = FALSE;
					if (!gbQuietMode) printf("Setting the maximum attenuation (RF OFF) for device %d\n", iDev + 1);
				}
				else
				{
					bTemp = TRUE;
					if (!gbQuietMode) printf("Setting the minimum attenuation (RF ON) for device %d\n", iDev + 1);
				}

				itemp = fnLDA_SetRFOn(MyDevices[iDev], bTemp);
				CheckAPISet(itemp);
			}

			// if we have a V2 Lab Brick, send it the additional commands
			if (fnLDA_GetFeatures(MyDevices[iDev]) > 0)
			{
				if (gbWantSetAStep2)
				{
					ftemp = (float)AStep2 / 4;
					if (!gbQuietMode) printf("Setting the second phase attenuation step for device %d to %.2f db \n", iDev + 1, ftemp);
					itemp = fnLDA_SetAttenuationStepTwo(MyDevices[iDev], AStep2);
					CheckAPISet(itemp);
				}
				if (gbWantSetDwell2)
				{
					if (!gbQuietMode) printf("Setting the second phase dwell time for device %d to %d ms \n", iDev + 1, Dwell2);
					itemp = fnLDA_SetDwellTimeTwo(MyDevices[iDev], Dwell2);
					CheckAPISet(itemp);
				}
				if (gbWantSetHold)
				{
					if (!gbQuietMode) printf("Setting the hold time between ramp phases for device %d to %d ms \n", iDev + 1, HoldTime);
					itemp = fnLDA_SetHoldTime(MyDevices[iDev], HoldTime);
					CheckAPISet(itemp);
				}
			}

			if (gbWantSaveSettings)
			{
				if (!gbQuietMode) printf("Saving the settings for device %d\n", iDev + 1);
				fnLDA_SaveSettings(MyDevices[iDev]);
				CheckAPISet(itemp);
			}

		}	// this is the end of our for loop over devices for the general commands

		// -- For ramps we first set the parameters, then send the actual commands to start the ramps
		//	  grouping these commands reduces the latency between the ramps on each attenuator
		for (iDev = DevNum; iDev < DevNum + DevRange; iDev++)
		{
			if (gbWantStartSweep)
			{
				// --- first we'll figure out what the user wants us to do ---

				if (Sweep_mode == 0)
				{
					if (!gbQuietMode) printf("Stopping the Attenuation Ramp\n");
					itemp = fnLDA_StartRamp(MyDevices[iDev], FALSE);
					CheckAPISet(itemp);

				}
				else
				{

					// --- The user wants to start some kind of an attenuation ramp ---
					if (Sweep_mode & CL_SWP_DIRECTION)
					{
						bTemp = FALSE;
					}
					else
					{
						bTemp = TRUE;
					}	// NB -- the flag is TRUE for "up" in the Set...Direction call.
					// but the old test program uses a 0 bit for up, and a 1 bit for down...

					itemp = fnLDA_SetRampDirection(MyDevices[iDev], bTemp);
					CheckAPISet(itemp);

					// --- and now we'll do the mode - one time or repeated ---
					if (Sweep_mode & CL_SWP_ONCE)
					{
						bTemp = FALSE;
					}
					else
					{
						bTemp = TRUE;
					}	// NB -- the flag is TRUE for "repeated" in the SetSweepMode call.
					// but the old test program encodes the modes differently

					itemp = fnLDA_SetRampMode(MyDevices[iDev], bTemp);
					CheckAPISet(itemp);

					// --- and then the bidirectional ramp control if the device is a V2 device
					if (fnLDA_GetFeatures(MyDevices[iDev]) > 0)
					{
						if (Sweep_mode & CL_SWP_BIDIRECTIONALLY)
						{
							bTemp = TRUE;
						}							// the command line has true for bidirectional 
						else						// as does the actual HW command...
						{
							bTemp = FALSE;
						}

						printf("Bidirection mode set to %x \n", bTemp);
						itemp = fnLDA_SetRampBidirectional(MyDevices[iDev], bTemp);
						CheckAPISet(itemp);
					}

					if (!gbQuietMode) printf("Starting an attenuation ramp for device %d\n", iDev + 1);
					itemp = fnLDA_StartRamp(MyDevices[iDev], TRUE);
					CheckAPISet(itemp);
				}
			}
		} // this is the end of our for loop over selected devices for the ramp command


		// -- Lets report on the device's operation for a little while, unless we are in batch mode

		if (!gbBatchMode)
		{
			j = 0;
			while (j < 40)
			{
				for (iDev = DevNum; iDev < DevNum + DevRange; iDev++)
				{
					// use the HiRes function and show all the channels for the device
					NumChannels = fnLDA_GetNumChannels(MyDevices[iDev]);
					if (NumChannels <= 0) NumChannels = 1;	// protect against an error return
					for (k = 1; k < NumChannels + 1; k++)
					{
						fnLDA_SetChannel(MyDevices[iDev], k);
						ftemp = ((float)fnLDA_GetAttenuationHR(MyDevices[iDev])) / 20;
						printf("Attenuation = %.2f db for device %d, channel %d\n", ftemp, iDev + 1, k);
					}
				}
				printf("\n");
				Sleep(500);		// wait for 1/2 second
				j++;
			}

		} // end of if not batch mode

		// -- we've done whatever the user wanted, time to close the devices
		for (j = 0; j < i; j++)
		{
			itemp = fnLDA_CloseDevice(MyDevices[j]);
		}

	} // end of if ( i > 0 ) -- "we have a device"

	return 0;
}

// ===================== end of main ======================================