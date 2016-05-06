//---------------------------------------------------------------------------
#ifndef dax22000_lib_DLL32H
#define dax22000_lib_DLL32H
//---------------------------------------------------------------------------

#define DWORD unsigned long int    // Use any of these if not defined.
#define WORD unsigned short int    // 
#define BYTE unsigned char         //
#define PVOID void *               //

//---------------------------------------------------------------------------
//  USER ROUTINES
//---------------------------------------------------------------------------

#define IMPORT extern "C" __declspec(dllimport)

IMPORT int DAx22000_GetNumCards(void);
IMPORT int DAx22000_Open(int CardNum);
IMPORT int DAx22000_Close(int CardNum);

IMPORT int DAx22000_Initialize(int CardNum);

IMPORT double DAx22000_SetClkRate(int CardNum, double User_Freq);
IMPORT int DAx22000_SelExtTrig(int CardNum, bool ExtTrig);

IMPORT int DAx22000_Run(int CardNum, bool TriggerNow);
IMPORT int DAx22000_Stop(int CardNum);

IMPORT int DAx22000_SoftTrigger(int CardNum);
IMPORT int DAx22000_Place_MRK2(int CardNum, int Mod16_CNT);

IMPORT int DAx22000_CreateSingleSegment(
   DWORD CardNum, 
   DWORD ChanNum, 
   DWORD NumPoints, 
   DWORD NumLoops, 
   DWORD PAD_Val_Beg, 
   DWORD PAD_Val_End,
   PVOID pUserArrayWORD, 
   DWORD Triggered
   );

IMPORT int DAx22000_CreateSegments(
   DWORD CardNum, 
   DWORD ChanNum, 
   DWORD NumSegments, 
   DWORD PAD_Val_Beg, 
   DWORD PAD_Val_End,
   PVOID pSegmentsList,
   bool Loop
   );
   
IMPORT int DAx22000_Debug(int CardNum, int ModeNum);         

IMPORT int DAx22000_Ext10MHz(int CardNum, int Enable);

#endif
