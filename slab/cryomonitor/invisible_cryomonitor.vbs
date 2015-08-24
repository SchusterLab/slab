Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "C:\_Lib\slab\cryomonitor\cryomonitor.bat" & Chr(34), 0
Set WshShell = Nothing