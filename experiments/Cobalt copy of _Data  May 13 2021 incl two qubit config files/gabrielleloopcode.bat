@echo off
SET /A "index = 1"
SET /A "count = 5"
:while
if %index% leq %count% (
   echo NEXT BATCH LOOP
   "C:\Users\Administrator\Anaconda3\python.exe" "S:\_Data\200507 - Test Continous Monitoring Picollo\run_cont_test0.py"
   goto :while
) (edited) 


