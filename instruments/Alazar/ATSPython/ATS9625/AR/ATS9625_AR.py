from __future__ import division
import numpy as np
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'Library'))
import atsapi as ats

samplesPerSec = None

# Configures a board for acquisition
def ConfigureBoard(board):
    # TODO: Select clock parameters as required to generate this
    # sample rate
    #
    # For example: if samplesPerSec is 100e6 (100 MS/s), then you can
    # either:
    #  - select clock source INTERNAL_CLOCK and sample rate
    #    SAMPLE_RATE_100MSPS
    #  - or select clock source FAST_EXTERNAL_CLOCK, sample rate
    #    SAMPLE_RATE_USER_DEF, and connect a 100MHz signal to the
    #    EXT CLK BNC connector
    global samplesPerSec
    samplesPerSec = 10000000.0
    board.setCaptureClock(ats.INTERNAL_CLOCK,
                          ats.SAMPLE_RATE_10MSPS,
                          ats.CLOCK_EDGE_RISING,
                          0)
    
    # TODO: Select channel A input parameters as required.
    board.inputControl(ats.CHANNEL_A,
                       ats.DC_COUPLING,
                       ats.INPUT_RANGE_PM_1_V_25,
                       ats.IMPEDANCE_50_OHM)
    
    # TODO: Select channel A bandwidth limit as required.
    board.setBWLimit(ats.CHANNEL_A, 0)
    
    
    # TODO: Select channel B input parameters as required.
    board.inputControl(ats.CHANNEL_B,
                       ats.DC_COUPLING,
                       ats.INPUT_RANGE_PM_1_V_25,
                       ats.IMPEDANCE_50_OHM)
    
    # TODO: Select channel B bandwidth limit as required.
    board.setBWLimit(ats.CHANNEL_B, 0)
    
    # TODO: Select trigger inputs and levels as required.
    board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                              ats.TRIG_ENGINE_J,
                              ats.TRIG_CHAN_A,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              150,
                              ats.TRIG_ENGINE_K,
                              ats.TRIG_DISABLE,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              128)

    # TODO: Select external trigger parameters as required.
    board.setExternalTrigger(ats.DC_COUPLING,
                             ats.ETR_5V)

    # TODO: Set trigger delay as required.
    triggerDelay_sec = 0
    triggerDelay_samples = int(triggerDelay_sec * samplesPerSec + 0.5)
    board.setTriggerDelay(triggerDelay_samples)

    # TODO: Set trigger timeout as required.
    #
    # NOTE: The board will wait for a for this amount of time for a
    # trigger event.  If a trigger event does not arrive, then the
    # board will automatically trigger. Set the trigger timeout value
    # to 0 to force the board to wait forever for a trigger event.
    #
    # IMPORTANT: The trigger timeout value should be set to zero after
    # appropriate trigger parameters have been determined, otherwise
    # the board may trigger if the timeout interval expires before a
    # hardware trigger event arrives.
    triggerTimeout_sec = 0
    triggerTimeout_clocks = int(triggerTimeout_sec / 10e-6 + 0.5)
    board.setTriggerTimeOut(triggerTimeout_clocks)

    # Configure AUX I/O connector as required
    board.configureAuxIO(ats.AUX_OUT_TRIGGER,
                         0)

def AcquireData(board, waitBar):
    # TODO: Select the number of pre-trigger samples
    preTriggerSamples = 1024

    # TODO: Select the number of samples per record.
    postTriggerSamples = 1024

    # TODO: Select the number of records in the acquisition.
    recordsPerCapture = 100

    # TODO: Select the amount of time to wait for the acquisition to
    # complete to on-board memory.
    acquisition_timeout_sec = 10

    # TODO: Select the active channels.
    channels = ats.CHANNEL_A | ats.CHANNEL_B
    channelCount = 0
    for c in ats.channels:
        channelCount += (c & channels == c)

    # TODO: Should data be saved to file?
    saveData = false
    dataFile = None
    if saveData:
        dataFile = open(os.path.join(os.path.dirname(__file__), "data.bin"), 'w')

    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + postTriggerSamples
    bytesPerRecord = bytesPerSample * samplesPerRecord

    # Calculate the size of a record buffer in bytes. Note that the
    # buffer must be at least 16 bytes larger than the transfer size.
    bytesPerBuffer = bytesPerSample * (samplesPerRecord + 0)

    # Set the record size
    board.setRecordSize(preTriggerSamples, postTriggerSamples)

    # Configure the number of records in the acquisition
    board.setRecordCount(recordsPerCapture)

    start = time.clock() # Keep track of when acquisition started
    board.startCapture() # Start the acquisition
    print("Capturing %d record. Press any key to abort" % recordsPerCapture)
    buffersCompleted = 0
    bytesTransferred = 0
    while not waitBar.hasUserCancelled():
        if not board.busy():
            # Acquisition is done
            break
        if time.clock() - start > acquisition_timeout_sec:
            board.abortCapture()
            raise Exception("Error: Capture timeout. Verify trigger")
        time.sleep(10e-3)

    captureTime_sec = time.clock() - start
    recordsPerSec = 0
    if captureTime_sec > 0:
        recordsPerSec = recordsPerCapture / captureTime_sec
    print("Captured %d records in %f rec (%f records/sec)" % (recordsPerCapture, captureTime_sec, recordsPerSec))

    buffer = ats.DMABuffer(bytesPerSample, bytesPerBuffer)

    # Transfer the records from on-board memory to our buffer
    print("Transferring %d records..." % recordsPerCapture)

    for record in range(recordsPerCapture):
        for channel in range(channelCount):
            channelId = ats.channels[channel]
            if channelId & channels == 0:
                continue
            board.read(channelId,             # Channel identifier
                       buffer.addr,           # Memory address of buffer
                       bytesPerSample,        # Bytes per sample
                       record + 1,            # Record (1-indexed)
                       -preTriggerSamples,    # Pre-trigger samples
                       samplesPerRecord)      # Samples per record
            bytesTransferred += bytesPerRecord;

            # Records are arranged in the buffer as follows:
            # R0A, R1A, R2A ... RnA, R0B, R1B, R2B ...
            #
            #
            # Sample codes are unsigned by default. As a result:
            # - a sample code of 0x0000 represents a negative full scale input signal.
            # - a sample code of 0x8000 represents a ~0V signal.
            # - a sample code of 0xFFFF represents a positive full scale input signal.

            # Optionaly save data to file
            if dataFile:
                buffer.buffer[:samplesPerRecord].tofile(dataFile)
        if waitBar.hasUserCancelled():
            break
        waitBar.setProgress(record / recordsPerCapture)

    # Compute the total transfer time, and display performance information.
    transferTime_sec = time.clock() - start
    bytesPerSec = 0
    if transferTime_sec > 0:
        bytesPerSec = bytesTransferred / transferTime_sec
    print("Transferred %d bytes (%f bytes per sec)" % (bytesTransferred, bytesPerSec))

# Handler for the SIGINT signal. This gets called when the user
# presses Ctrl+C.
def sigint_handler(signal, frame):
    print("Acquisition aborting. Please wait.")
    abortAcquisition = True
# Register SIGINT handler
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == "__main__":
    board = ats.Board(systemId = 1, boardId = 1)
    waitBar = ats.WaitBar()
    ConfigureBoard(board)
    AcquireData(board, waitBar)