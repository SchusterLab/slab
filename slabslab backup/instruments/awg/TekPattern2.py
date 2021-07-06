"""
TekPattern2 for creating Tek70001 AWG files. 

Created on April 10, 2014

Dave M
"""

import struct
import io
import h5py
import numpy as np
import io


#This creates a waveform file that can be loaded into the awg
def create_waveform_file(filename,data):
    
    
    data = data.astype(np.float32)   
    
    #works if you specify single, floating and then each point is specified by a floating #
    #from -1 --> 1
    
    #datafile offset is where the data starts
    #first write to a string to determine the data offset!
    for i in range(2):
        if i==0:
            FID = io.StringIO()
        else:
            str_length = len(FID.getvalue())
            FID = io.open(filename, 'wb')
        
        if i==0:
            FID.write("<DataFile offset=\"000000000\" version=\"0.1\">")
        else:
            FID.write("<DataFile offset=\""+"{:09d}".format(str_length)+ "\" version=\"0.1\">")
        FID.write("<DataSetsCollection xmlns=\"http://www.tektronix.com\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.tektronix.com file:///C:\\Program%20Files\\Tektronix\\AWG70000\\AWG\\Schemas\\awgDataSets.xsd\">")
        FID.write("<DataSets version=\"1\" xmlns=\"http://www.tektronix.com\">")
        FID.write("<DataDescription>")
        FID.write("<NumberSamples>" + str(len(data)) + "</NumberSamples>")
        FID.write("<SamplesType>AWGWaveformSample</SamplesType>")
        FID.write("<MarkersIncluded>false</MarkersIncluded>")
        
        #number formats: Single, UInt16, Int32, Double
        FID.write("<NumberFormat>Single</NumberFormat>")
        FID.write("<Endian>Big</Endian>")
        FID.write("<Timestamp>2014-04-01T16:29:23.8235574-07:00</Timestamp>")
        FID.write("</DataDescription>")
        FID.write("<ProductSpecific name=\"\">")
        FID.write("<ReccSamplingRate units=\"Hz\">50000000000</ReccSamplingRate>")
        FID.write("<ReccAmplitude units=\"Volts\">1.0</ReccAmplitude>")
        FID.write("<ReccOffset units=\"Volts\">0</ReccOffset>")
        FID.write("<SerialNumber />")
        FID.write("<SoftwareVersion>2.0.0211</SoftwareVersion>")
        FID.write("<UserNotes />")
        
        #Floating, EightBit, NineBit, TenBit (What do these mean?)
        FID.write("<OriginalBitDepth>Floating</OriginalBitDepth>")
        FID.write("<Thumbnail />")
        FID.write("<CreatorProperties name=\"\" />")
        FID.write("  </ProductSpecific>")
        FID.write("</DataSets>")
        FID.write("</DataSetsCollection>")
        FID.write("<Setup />")
        FID.write("</DataFile>")  
    
   
    FID.write(data.tostring())
    
    FID.close()
    
    
if __name__ == '__main__':

    pass
        
    
