This release of the x64 version of the Vaunix Lab Brick Phase Shifter DLL contains a test program which can
be built using Visual Studio 2013. Only the x64 configuration of the test program is supported by this SDK.

The DLL supports the LPS-402, LPS-802, and LPS-123 phase shifters.

The DLL has ANSI-C style naming, but uses the standard x64 calling sequence. The ANSI-C style naming was chosen to simplify the task of accessing
the DLL functions from other languages such as Python or Tcl.

The DLL relies on run time libraries from the current versions of Windows, and therefore may require installation of the Visual Studio 2013 
redistributable libraries on some systems.

Released 6/13/2014