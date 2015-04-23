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
                       ats.INPUT_RANGE_PM_400_MV,
                       ats.IMPEDANCE_50_OHM)
    
    
    # TODO: Select channel B input parameters as required.
    board.inputControl(ats.CHANNEL_B,
                       ats.DC_COUPLING,
                       ats.INPUT_RANGE_PM_400_MV,
                       ats.IMPEDANCE_50_OHM)
    
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

def AcquireData(boards, acquireData):
    
    # No pre-trigger samples in NPT mode
    preTriggerSamples = 0

    # TODO: Select the number of samples per record.
    postTriggerSamples = 2048

    # TODO: Select the number of records per DMA buffer.
    recordsPerBuffer = 10

    # TODO: Select the number of buffers per acquisition.
    buffersPerAcquisition = 10
    
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

    # Make sure that boards[0] is the system's master
    if boards[0].boardId != 1:
        raise ValueError("The first board passed should be the master.")
    for board in boards:
        if board.systemId != boards[0].systemId:
            raise ValueError("All the boards should be of the same system.")

    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = boards[0].getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + postTriggerSamples
    bytesPerRecord = bytesPerSample * samplesPerRecord
    bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount

    # TODO: Select number of DMA buffers to allocate
    bufferCount = 4

    buffers = []
    for b in range(len(boards)):
        # Allocate DMA buffers
        buffers.append([])
        for i in range(bufferCount):
            buffers[b].append(ats.DMABuffer(bytesPerSample, bytesPerBuffer))

        
        # Set the record size
        boards[b].setRecordSize(preTriggerSamples, postTriggerSamples)

        recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition

        # Configure the board to make an NPT AutoDMA acquisition
        boards[b].beforeAsyncRead(channels,
                                  -preTriggerSamples,
                                  samplesPerRecord,
                                  recordsPerBuffer,
                                  recordsPerAcquisition,
                                  ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT | ats.ADMA_FIFO_ONLY_STREAMING)
        


        # Post DMA buffers to board
        for buffer in buffers[b]:
            boards[b].postAsyncBuffer(buffer.addr, buffer.size_bytes)

    start = time.clock() # Keep track of when acquisition started
    boards[0].startCapture() # Start the acquisition
    buffersPerAcquisitionAllBoards = len(boards) * buffersPerAcquisition
    print("Capturing %d buffers. Press any key to abort" % buffersPerAcquisitionAllBoards)
    buffersCompletedPerBoard = 0
    buffersCompletedAllBoards = 0
    bytesTransferredAllBoards = 0
    while buffersCompletedPerBoard < buffersPerAcquisition and not waitBar.hasUserCancelled():
        for b in range(len(boards)):
            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            buffer = buffers[b][buffersCompletedPerBoard % len(buffers[b])]
            boards[b].waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
            buffersCompletedAllBoards += 1
            bytesTransferredAllBoards += buffer.size_bytes

            # TODO: Process sample data in this buffer. Data is available
            # as a NumPy array at buffer.buffer

            # NOTE:
            #
            # While you are processing this buffer, the board is already
            # filling the next available buffer(s).
            #
            # You MUST finish processing this buffer and post it back to the
            # board before the board fills all of its available DMA buffers
            # and on-board memory.
            #
            # Samples are arranged in the buffer as follows: S0A, S0B, ..., S1A, S1B, ...
            # with SXY the sample number X of channel Y.
            #
            # A 12-bit sample code is stored in the most significant bits of
            # in each 16-bit sample value.
            #
            # Sample codes are unsigned by default. As a result:
            # - a sample code of 0x0000 represents a negative full scale input signal.
            # - a sample code of 0x8000 represents a ~0V signal.
            # - a sample code of 0xFFFF represents a positive full scale input signal.
            # Optionaly save data to file
            if dataFile:
                buffer.buffer.tofile(dataFile)

            # Add the buffer to the end of the list of available buffers.
            boards[b].postAsyncBuffer(buffer.addr, buffer.size_bytes)
        buffersCompletedPerBoard += 1
        # Update progress bar
        waitBar.setProgress(buffersCompletedPerBoard / buffersPerAcquisition)


    # Compute the total transfer time, and display performance information.
    transferTime_sec = time.clock() - start
    print("Capture completed in %f sec" % transferTime_sec)
    buffersPerSec = 0
    bytesPerSec = 0
    recordsPerSec = 0
    if transferTime_sec > 0:
        buffersPerSec = buffersCompletedAllBoards / transferTime_sec
        bytesPerSec = bytesTransferredAllBoards / transferTime_sec
        recordsPerSec = recordsPerBuffer * buffersCompletedAllBoards / transferTime_sec
    print("Captured %d buffers (%f buffers per sec)" % (buffersCompletedAllBoards, buffersPerSec))
    print("Captured %d records (%f records per sec)" % (recordsPerBuffer * buffersCompletedAllBoards, recordsPerSec))
    print("Transferred %d bytes (%f bytes per sec)" % (bytesTransferredAllBoards, bytesPerSec))

    # Abort transfer.
    board.abortAsyncRead()

# Handler for the SIGINT signal. This gets called when the user
# presses Ctrl+C.
def sigint_handler(signal, frame):
    print("Acquisition aborting. Please wait.")
    abortAcquisition = True
# Register SIGINT handler
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == "__main__":
    boards = []
    systemId = 1
    for i in range(ats.boardsInSystemBySystemID(systemId)):
        boards.append(ats.Board(systemId, i + 1))
    waitBar = ats.WaitBar()
    for board in boards:
        ConfigureBoard(board)
    AcquireData(boards, waitBar)