// LPSTest_64.cpp -- A test shell for the Vaunix LPS phase shifter interface dlls
//
//	RD 6/2014 -- 64 bit demo program 
//
//	V.9 

#include "stdafx.h"
#include "vnx_lps_api.h"

// ------------------------------- Allocations -------------------------------------------

static DEVID MyDevices[MAXDEVICES];			// I have statically allocated this array for convenience
// It holds a list of device IDs for the connected devices
// They are stored starting at MyDevices[0]

static char MyDeviceName[MAX_MODELNAME];		// NB -- this is a single byte char array for testing the ASCII name function
static wchar_t MyDeviceNameW[MAX_MODELNAME];	// NB -- this is a WCHAR array for testing the Unicode name function

static wchar_t errmsg[32];						// For the status->string converter
static char cModelName[32];						// buffer for the model name


static string sDevName = "LPS-802";				// device name string
static bool gbWantOneDevice = FALSE;

static long DevNum = 0;							// which device we should work with

// --------- NB: a lot of these default values are never used ------------
static string sDevFrequency = "";
static long Frequency = 10000;					// default test frequency 1GHz in 100KHz units

static string sDevStart = "";
static long Start = 0;							// default ramp start is 0 degrees

static string sDevStop = "";
static long Stop = 90;							// default ramp stop is 90 degrees

static string sPhaseStep = "";
static long PhaseStep = 1;						// default phase step is 1 degree

static string sPhaseStep2 = "";
static long PhaseStep2 = 1;						// default phase step is 1 degree

static string sDevSweep = "";
static long DwellTime = 1000;					// default dwell time for section one is 1 second

static string sDevSweep2 = "";
static long DwellTime2 = 100;					// default dwell time for section two is .1 second

static string sDevIdle = "";
static long IdleTime = 200;						// default idle between ramps is .2 second

static string sDevHold = "";
static long HoldTime = 10;						// default hold time between section one and two is .01 second

static string sDevPhase = "";
static long PhaseAngle = 0;						// default phase angle of zero degrees

static int Sweep_mode = 0;						// a variable to hold the user's desired sweep mode
static int Profile_mode = 0;					// a variable to hold the user's desired profile mode

static int Profile_index = 0;
static int Profile_value = 0;					// index and value for setting profile elements

bool gbWantSetFrequency = FALSE;
bool gbWantSetStart = FALSE;
bool gbWantSetStop = FALSE;
bool gbWantSetPhaseStep = FALSE;
bool gbWantSetPhaseStep2 = FALSE;
bool gbWantSetDwellTime = FALSE;
bool gbWantSetDwellTime2 = FALSE;
bool gbWantSetIdleTime = FALSE;
bool gbWantSetHoldTime = FALSE;
bool gbWantStartSweep = FALSE;
bool gbWantSetPhase = FALSE;

bool gbWantSetElement = FALSE;
bool gbWantStartProfile = FALSE;

bool gbWantSaveSettings = FALSE;

bool gbGotReply = FALSE;
bool gbBatchMode = FALSE;


// ------------------------------- Support Routines --------------------------------------

void PrintHelp()
{
	printf("LPS Test\n");
	printf("\n");
	printf("Hit CTRL+C to exit\n");
	printf("\n");

	printf(" --- Overall modes and device selection. Defaults to all devices ---\n");
	printf(" This version controls the first Lab Brick Digital Phase Shifter it finds\n");
	printf(" Each command can only appear once on the command line\n");
	printf("  -b        Batch mode, exit after sending device commands\n");
	printf("  -d        Device Number -- 1 to NDevices\n");
	printf("\n");

	printf(" --- Commands to set parameters and start sweep --- \n");

	printf("  -f nn     Set working frequency, nn is frequency in 1 Hz (1.0e9 for 1GHz) units\n");
	printf("  -p nn     Set phase angle, nn is in degrees,\n");
	printf("  -s nn     Set phase ramp start value, nn is start phase shift in degrees\n");
	printf("  -e nn     Set phase ramp end value, nn is end phase shift in degrees\n");
	printf("  -a nn     Set the phase angle change at each step in section one, nn is in degrees.\n");
	printf("  -c nn     Set the phase angle change at each step in section two, nn is in degrees.\n");
	printf("  -t nn     Set the dwell time at each step in section one, or in a profile, nn is time in ms.\n");
	printf("  -r nn     Set the dwell time at each step in section two, nn is time in ms.\n");
	printf("  -i nn		Set the idle time between ramps or profiles, nn is time in ms.\n");
	printf("  -h nn		Set the hold time between ramp section one and two, nn is time in ms.\n");

	printf("  -g n      Start a ramp, 1 = once upwards, 2 = continuous upwards\n");
	printf("             5 = once down, 6 = continuous down, 0 = end sweep\n");
	printf("             17 = single bi-directional sweep, 18 = continuous bi-directional sweep\n");
	printf("\n");
	printf("  -v i nn	Set an element in the profile, i is the index, 0 to 49 and nn is the phase angle\n");
	printf("  -x n      Start a profile, 1 = once, 2 = repeating, 0 to stop\n");

	printf("  -y        Write user settings to flash\n");
	printf("\n");


}

// --------------------- MakeLower ------------------------------

wchar_t MakeLowerW(wchar_t &wc)
{
	return wc = towlower(wc);
}

// --------------------------------------------------------------

#define MAX_MSG 32

/* A function to display the status as string */
wchar_t* fnLMS_perror(LPSTATUS status)
{
		wcscpy_s(errmsg, MAX_MSG, L"STATUS_OK");
		if (BAD_PARAMETER == status) wcscpy_s(errmsg, MAX_MSG, L"BAD_PARAMETER");
		if (BAD_HID_IO == status) wcscpy_s(errmsg, MAX_MSG, L"BAD_HID_IO");
		if (DEVICE_NOT_READY == status) wcscpy_s(errmsg, MAX_MSG, L"DEVICE_NOT_READY");

		// Status returns for DevStatus
		if (INVALID_DEVID == status) wcscpy_s(errmsg, MAX_MSG, L"INVALID_DEVID");
		if (DEV_CONNECTED == status) wcscpy_s(errmsg, MAX_MSG, L"DEV_CONNECTED");
		if (DEV_OPENED == status) wcscpy_s(errmsg, MAX_MSG, L"DEV_OPENED");
		if (SWP_ACTIVE == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_ACTIVE");
		if (SWP_UP == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_UP");
		if (SWP_REPEAT == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_REPEAT");
		if (SWP_BIDIRECTIONAL == status) wcscpy_s(errmsg, MAX_MSG, L"SWP_BIDIRECTIONAL");

		return errmsg;

}

// ---------- ParseCommandLine ----------------------------------------------- 

// ParseCommandLine() will return FALSE to indicate that we received an invalid
// command or should abort for another reason.

bool ParseCommandLine(int argc, _TCHAR *argv[])
{
	enum {
		wantDash, wantDevNumber, wantFrequency, wantStart, wantStop, wantPhaseStep, wantPhaseStep2, wantDwell,
		wantDwell2, wantIdle, wantPhase, wantHold, wantSweep, wantProfileMode, wantProfileIndex,
		wantProfileValue
	} state = wantDash;

	for (int i = 1; i < argc; ++i) {
		// Convert each argument to lowercase
		wstring thisParam(argv[i]);
		for_each(thisParam.begin(), thisParam.end(), MakeLowerW);

		// if we're looking for a command, handle the - before the command letter
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

			// Identify the argumenets
			if (L"d" == thisParam) {
				// -d should be followed by a number
				state = wantDevNumber;
			}
			else if (L"b" == thisParam) {
				gbBatchMode = TRUE;
			}
			else if (L"f" == thisParam) {
				gbWantSetFrequency = TRUE;
				state = wantFrequency;
			}
			else if (L"p" == thisParam) {
				gbWantSetPhase = TRUE;
				state = wantPhase;
			}
			else if (L"s" == thisParam) {
				gbWantSetStart = TRUE;
				state = wantStart;
			}
			else if (L"e" == thisParam) {
				gbWantSetStop = TRUE;
				state = wantStop;
			}
			else if (L"a" == thisParam) {
				gbWantSetPhaseStep = TRUE;
				state = wantPhaseStep;
			}
			else if (L"c" == thisParam) {
				gbWantSetPhaseStep2 = TRUE;
				state = wantPhaseStep2;
			}
			else if (L"t" == thisParam) {
				gbWantSetDwellTime = TRUE;
				state = wantDwell;
			}
			else if (L"r" == thisParam) {
				gbWantSetDwellTime2 = TRUE;
				state = wantDwell2;
			}
			else if (L"i" == thisParam) {
				gbWantSetIdleTime = TRUE;
				state = wantIdle;
			}
			else if (L"h" == thisParam) {
				gbWantSetHoldTime = TRUE;
				state = wantHold;
			}
			else if (L"g" == thisParam) {
				gbWantStartSweep = TRUE;
				state = wantSweep;
			}
			else if (L"x" == thisParam) {
				gbWantStartProfile = TRUE;
				state = wantProfileMode;
			}
			else if (L"v" == thisParam) {
				gbWantSetElement = TRUE;
				state = wantProfileIndex;
			}
			else if (L"y" == thisParam) {
				gbWantSaveSettings = TRUE;
				state = wantDash;
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

			case wantDevNumber:
				DevNum = _wtoi(thisParam.c_str());
				state = wantDash;	// we always go back to the wantDash state to look for the next arg.
				break;

			case wantFrequency:
				Frequency = (int)(wcstof(thisParam.c_str(), NULL) / 100000);		// convert to a float first...
																					// then scale to 100KHz units
				state = wantDash;
				break;

			case wantStart:
				Start = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantStop:
				Stop = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantPhaseStep:
				PhaseStep = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantPhaseStep2:
				PhaseStep2 = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantDwell:
				DwellTime = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantDwell2:
				DwellTime2 = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantPhase:
				PhaseAngle = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantIdle:
				IdleTime = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantHold:
				HoldTime = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantSweep:
				Sweep_mode = _wtoi(thisParam.c_str());
				state = wantDash;
				break;

			case wantProfileMode:
				Profile_mode = _wtoi(thisParam.c_str());
				state = wantDash;
				break;
			case wantProfileIndex:
				Profile_index = _wtoi(thisParam.c_str());
				state = wantProfileValue;
				break;
			case wantProfileValue:
				Profile_value = _wtoi(thisParam.c_str());
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


int _tmain(int argc, _TCHAR* argv[])
{
	int i, j;
	int itemp, itemp2;
	bool bTemp;
	float ftemp;


	printf("Lab Brick Digital Phase Shifter Test Program\n");

	if (!ParseCommandLine(argc, argv))
		return 0;

	DevNum = DevNum - 1;
	if (DevNum < 0) DevNum = 0;


	// --- if TestMode = TRUE then the dll will fake the hardware ---
	fnLPS_SetTestMode(FALSE);

	i = fnLPS_GetNumDevices();

	if (i == 0){
		printf("No device found\n");
	}

	if (i == 1){
		printf("Found %d Device\n", i);

	}
	else {
		printf("Found %d Devices\n", i);
	}

	i = fnLPS_GetDevInfo(MyDevices);

	printf("Got Device Info for %d Device[s]\n", i);


	if (i > 0)	// do we have a device?
	{
		for (j = 0; j < i; j++){

			// --- print out the first device's name ---
			itemp = fnLPS_GetModelNameA(MyDevices[j], MyDeviceName);
			printf("Device %d is an %s \n", i, MyDeviceName);

			// --- print out the device's serial number ---
			itemp = fnLPS_GetSerialNumber(MyDevices[j]);
			printf("Device %d has serial number %d \n", i, itemp);


			// --- We need to init the device (open it) before we can do anything else ---
			itemp = fnLPS_InitDevice(MyDevices[j]);

			if (itemp){
				printf("InitDevice returned %x\n", itemp);
			}

			// --- Lets see if we got the device's parameters ---

			itemp = fnLPS_GetMinWorkingFrequency(MyDevices[j]);
			printf("Minimum Working Frequency = %d (100KHz units)\n", itemp);

			itemp = fnLPS_GetMaxWorkingFrequency(MyDevices[j]);
			printf("Maximum Working Frequency = %d\n", itemp);

			itemp = fnLPS_GetWorkingFrequency(MyDevices[j]);
			printf("Working Frequency = %d Megahertz\n", itemp / 10);

			itemp = fnLPS_GetPhaseAngle(MyDevices[j]);
			printf("Phase Shift = %d degrees\n", itemp);

			// --- Show the ramp parameters ---

			printf(" Ramp Parameters: \n");
			itemp = fnLPS_GetRampStart(MyDevices[j]);
			itemp2 = fnLPS_GetRampEnd(MyDevices[j]);
			printf("Ramp from %d degrees to %d degrees\n", itemp, itemp2);

			itemp = fnLPS_GetDwellTime(MyDevices[j]);
			itemp2 = fnLPS_GetPhaseAngleStep(MyDevices[j]);
			ftemp = (float)itemp / 1000;
			printf("First section phase step %d degrees every %f seconds\n", itemp2, ftemp);

			itemp = fnLPS_GetDwellTimeTwo(MyDevices[j]);
			itemp2 = fnLPS_GetPhaseAngleStepTwo(MyDevices[j]);
			ftemp = (float)itemp / 1000;
			printf("Second section phase step %d degrees every %f seconds\n", itemp2, ftemp);

			itemp = fnLPS_GetHoldTime(MyDevices[j]);
			ftemp = (float)itemp / 1000;
			printf("Wait for %f seconds at the end of the first section\n", ftemp);
			itemp = fnLPS_GetIdleTime(MyDevices[j]);
			ftemp = (float)itemp / 1000;
			printf("Wait for %f seconds before repeating the ramp\n", ftemp);

			printf(" --------------------------------- \n");

		} // end of the for loop over the devices

		// ------------- Now we'll set the requested device with new parameters -------------


		// --- Set the working frequency first ---

		if (gbWantSetFrequency)
		{

			printf("Setting Working Frequency = %d MHz\n", Frequency / 10);
			itemp = fnLPS_SetWorkingFrequency(MyDevices[DevNum], Frequency);
		}

		// --- and then do whatever else the user requested ---

		if (gbWantSetPhase)
		{
			printf("Setting Phase Angle = %d degrees\n", PhaseAngle);
			itemp = fnLPS_SetPhaseAngle(MyDevices[DevNum], PhaseAngle);
		}


		if (gbWantSetStart)
		{
			itemp = fnLPS_SetRampStart(MyDevices[DevNum], Start);
		}

		if (gbWantSetStop)
		{
			itemp = fnLPS_SetRampEnd(MyDevices[DevNum], Stop);
		}

		if (gbWantSetPhaseStep)
		{
			itemp = fnLPS_SetPhaseAngleStep(MyDevices[DevNum], PhaseStep);
		}

		if (gbWantSetPhaseStep2)
		{
			itemp = fnLPS_SetPhaseAngleStepTwo(MyDevices[DevNum], PhaseStep2);
		}

		if (gbWantSetDwellTime)
		{
			itemp = fnLPS_SetDwellTime(MyDevices[DevNum], DwellTime);
			itemp = fnLPS_SetProfileDwellTime(MyDevices[DevNum], DwellTime);
		}

		if (gbWantSetDwellTime2)
		{
			itemp = fnLPS_SetDwellTimeTwo(MyDevices[DevNum], DwellTime2);
		}

		if (gbWantSetIdleTime)
		{
			itemp = fnLPS_SetIdleTime(MyDevices[DevNum], IdleTime);
			itemp = fnLPS_SetProfileIdleTime(MyDevices[DevNum], IdleTime);
		}

		if (gbWantSetHoldTime)
		{
			itemp = fnLPS_SetHoldTime(MyDevices[DevNum], HoldTime);
		}

		if (gbWantStartSweep)
		{
			// --- first we'll figure out what the user wants us to do ---
			if (Sweep_mode & LPS_SWP_DIRECTION)
			{
				bTemp = FALSE;
			}
			else
			{
				bTemp = TRUE;
			}	// NB -- don't confuse these similarly named LPS_ constants for the API status constants!!


			itemp = fnLPS_SetRampDirection(MyDevices[DevNum], bTemp);	// TRUE means ramp upwards for the API

			// --- and now we'll do the mode - one time sweep or repeated sweep ---

			if (Sweep_mode & LPS_SWP_ONCE)
			{
				bTemp = FALSE;
			}
			else
			{
				bTemp = TRUE;
			}

			itemp = fnLPS_SetRampMode(MyDevices[DevNum], bTemp);		// TRUE means repeated sweep for the API


			if (Sweep_mode & LPS_SWP_BIDIR)
			{
				itemp = fnLPS_SetRampBidirectional(MyDevices[DevNum], TRUE);
			}
			else
			{
				itemp = fnLPS_SetRampBidirectional(MyDevices[DevNum], FALSE);
			}


			if (!Sweep_mode)
			{
				itemp = fnLPS_StartRamp(MyDevices[DevNum], FALSE);
			}
			else
			{
				printf("Starting a Ramp with mode = %x\n", Sweep_mode);
				itemp = fnLPS_StartRamp(MyDevices[DevNum], TRUE);
			}

		}

		// -------- profile related commands ---------
		if (gbWantStartProfile)
		{
			itemp = fnLPS_StartProfile(MyDevices[DevNum], Profile_mode);
		}

		if (gbWantSetElement)
		{
			itemp = fnLPS_SetProfileElement(MyDevices[DevNum], Profile_index, Profile_value);
		}


		// --- do this last, since the user probably wants to save what he just set ---

		if (gbWantSaveSettings)
		{
			fnLPS_SaveSettings(MyDevices[DevNum]);
		}

		// -- The user wants us to exit right away --

		if (gbBatchMode)
		{
			for (j = 0; j < i; j++)
			{
				itemp = fnLPS_CloseDevice(MyDevices[j]);

			}
			return 0;		// we're done, exit to the command prompt
		}


		// -- Lets hang around some and report on the device's operation

		j = 0;

		while (j < 20)
		{

			itemp = fnLPS_GetPhaseAngle(MyDevices[DevNum]);
			printf("Phase Shift = %d degrees\n", itemp);

			Sleep(500);		// wait for 1/2 second

			j++;

		}

		// -- we've done whatever the user wanted, time to close the devices

		printf("Closing devices...\n");

		for (j = 0; j < i; j++)
		{
			itemp = fnLPS_CloseDevice(MyDevices[j]);

		}

	} // end of if ( i > 0 ) -- "we have a device"

	return 0;
}

