@echo off
SET /A "index = 1"
SET /A "count = 5"
:while
if %index% leq %count% (
   "C:\ProgramData\Anaconda3\envs\py36vis\python.exe" "S:\_Data\210412 - PHMIV3_56 - BF4 cooldown 2\_cont_data_taking\run_cont.py"
   echo NEXT BATCH LOOP
   goto :while
)