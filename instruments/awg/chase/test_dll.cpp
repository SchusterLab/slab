#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <math.h>
//#include <windows.h>


#include "dax22000_lib_DLL32.h"

//using namespace std; // better to use std::cout, std::cin


int main(int argc, char** argv) 
{
   DWORD NumCards = 0;
   DWORD CardNum = 1;
   DWORD Chan = 1;
   int x;
   double Actual_Frequency;
   
   WORD TempArray[8000];
   DWORD MemoryDepth = 512;
   double pi = 3.14159265358979;

//------------------------------------------------------------------------------
// CREATE SAMPLE SINEWAVE TO UPLOAD
//------------------------------------------------------------------------------
   for (x=0; x < (MemoryDepth); x++) {
      TempArray[x] = (unsigned int) ( ceil( 2047.5 + 2047.5*sin( 2.0*pi* x/(32) ) ) );
   }

   std::cout<<"TempArray loaded with Sinewave.\nHit key to continue ...\n";
   std::cin.get();

//------------------------------------------------------------------------------
// CHECK IF CARD DETECTED; IF NOT THEN EXIT PROGRAM
//------------------------------------------------------------------------------
   NumCards = DAx22000_GetNumCards();

   std::cout << "Number of Cards Detected = " << NumCards << "\nHit key to continue ...\n";
   std::cin.get();

   if (NumCards != 1) exit(0);
   
//------------------------------------------------------------------------------
// OPEN DRIVER, INITIALIZE, SET CLOCK RATE
//------------------------------------------------------------------------------

   x = DAx22000_Open(1);
   
   std::cout << "DAx22000_Open = " << x << "\nHit key to continue ...\n";
   std::cin.get();
   
   x = DAx22000_Initialize(1);
   
   std::cout << "DAx22000_Initialize = " << x << "\nHit key to continue ...\n";
   std::cin.get();
   
   Actual_Frequency = DAx22000_SetClkRate(1, 2.0e9);
   
   std::cout << "DAx22000_SetClkRate = " << Actual_Frequency << "\nHit key to continue ...\n";
   std::cin.get();
    
     
//------------------------------------------------------------------------------
// UPLOAD USER WAVEFORM
//------------------------------------------------------------------------------
  
   x = DAx22000_CreateSingleSegment(
      1,           // DWORD CardNum
      1,           // DWORD ChanNum
      64,          // DWORD NumPoints, 
      0,           // DWORD NumLoops,        // 0 = Continuous Loop
      2047,        // DWORD PAD_Val_Beg, 
      2047,        // DWORD PAD_Val_End,
      TempArray,   // PVOID pUserArrayWORD, 
      1            // DWORD Triggered        // 1 = User can initiate "NumLoops" above by 
   );                                        //     triggering externally or SoftTrigger.
   
   std::cout << "DAx22000_CreateSingleSegment = " << x << "\nHit key to continue ...\n";
   std::cin.get();
   
   
//------------------------------------------------------------------------------
// OUTPUT DATA
//------------------------------------------------------------------------------

   DAx22000_Run(1, true);
   
   std::cout << "Outputing Data.\nHit key to close driver and shut off card.\n";
   std::cin.get();
   
//------------------------------------------------------------------------------
// STOP OUTPUT AND CLOSE DRIVER
//------------------------------------------------------------------------------

   DAx22000_Stop(1);
   
   DAx22000_Close(1);

   return 0;
}
