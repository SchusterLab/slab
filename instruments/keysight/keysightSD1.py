import os;
import sys
from ctypes import *
from math import pow, log, ceil
from abc import ABCMeta, abstractmethod
import slab.instruments.keysight.SD1constants as constants
import numpy as np

if sys.version_info[1] > 7 :
    os.add_dll_directory('C:\\Program Files\\Keysight\\SD1\\shared')
    os.add_dll_directory('C:\\Program Files\\Common Files\\Keysight\\PathWave Test Sync Executive\\HVIcore\\1.0\\bin')



def to_numpy_float(data):
	if type(data) != np.ndarray:
		return np.array(data, np.float)
	if data.dtype != np.float:
		return data.astype(np.float)
	return data

class SD_Object :
    __core_dll = cdll.LoadLibrary("SD1core" if os.name == 'nt' else "libSD1core.so")

    def __init__(self) :
        self.__handle = 0;

    @classmethod
    def __formatString(cls, string) :
        tmp = string.decode();
        return tmp[0:tmp.find('\0')];


class SD_Error(SD_Object) :
    STATUS_OK = 0;
    NONE = 0;
    STATUS_DEMO = 1;
    OPENING_MODULE = -8000;
    CLOSING_MODULE = -8001;
    OPENING_HVI = -8002;
    CLOSING_HVI = -8003;
    MODULE_NOT_OPENED = -8004;
    MODULE_NOT_OPENED_BY_USER = -8005;
    MODULE_ALREADY_OPENED = -8006;
    HVI_NOT_OPENED = -8007;
    INVALID_OBJECTID = -8008;
    INVALID_MODULEID = -8009;
    INVALID_MODULEUSERNAME = -8010;
    INVALID_HVIID = -8011;
    INVALID_OBJECT = -8012;
    INVALID_NCHANNEL = -8013;
    BUS_DOES_NOT_EXIST = -8014;
    BITMAP_ASSIGNED_DOES_NOT_EXIST = -8015;
    BUS_INVALID_SIZE = -8016;
    BUS_INVALID_DATA = -8017;
    INVALID_VALUE = -8018;
    CREATING_WAVE = -8019;
    NOT_VALID_PARAMETERS = -8020;
    AWG_FAILED = -8021;
    DAQ_INVALID_FUNCTIONALITY = -8022;
    DAQ_POOL_ALREADY_RUNNING = -8023;
    UNKNOWN = -8024;
    INVALID_PARAMETERS = -8025;
    MODULE_NOT_FOUND = -8026;
    DRIVER_RESOURCE_BUSY = -8027;
    DRIVER_RESOURCE_NOT_READY = -8028;
    DRIVER_ALLOCATE_BUFFER = -8029;
    ALLOCATE_BUFFER = -8030;
    RESOURCE_NOT_READY = -8031;
    HARDWARE = -8032;
    INVALID_OPERATION = -8033;
    NO_COMPILED_CODE = -8034;
    FW_VERIFICATION = -8035;
    COMPATIBILITY = -8036;
    INVALID_TYPE = -8037;
    DEMO_MODULE = -8038;
    INVALID_BUFFER = -8039;
    INVALID_INDEX = -8040;
    INVALID_NHISTOGRAM = -8041;
    INVALID_NBINS = -8042;
    INVALID_MASK = -8043;
    INVALID_WAVEFORM = -8044;
    INVALID_STROBE = -8045;
    INVALID_STROBE_VALUE = -8046;
    INVALID_DEBOUNCING = -8047;
    INVALID_PRESCALER = -8048;
    INVALID_PORT = -8049;
    INVALID_DIRECTION = -8050;
    INVALID_MODE = -8051;
    INVALID_FREQUENCY = -8052;
    INVALID_IMPEDANCE = -8053;
    INVALID_GAIN = -8054;
    INVALID_FULLSCALE = -8055;
    INVALID_FILE = -8056;
    INVALID_SLOT = -8057;
    INVALID_NAME = -8058;
    INVALID_SERIAL = -8059;
    INVALID_START = -8060;
    INVALID_END = -8061;
    INVALID_CYCLES = -8062;
    HVI_INVALID_NUMBER_MODULES = -8063;
    DAQ_P2P_ALREADY_RUNNING = -8064;
    OPEN_DRAIN_NOT_SUPPORTED = -8065;
    CHASSIS_PORTS_NOT_SUPPORTED = -8066;
    CHASSIS_SETUP_NOT_SUPPORTED = -8067;
    OPEN_DRAIN_FAILED = -8068;
    CHASSIS_SETUP_FAILED = -8069;
    INVALID_PART = -8070;
    INVALID_SIZE = -8071;
    INVALID_HANDLE = -8072;
    NO_WAVEFORMS_IN_LIST = -8073
    PATHWAVE_REGISTER_NOT_FOUND = -8074
    SD_ERROR_HVI_DRIVER_ERROR = -8075
    BAD_MODULE_OPEN_OPTION = -8076
    NOT_HVI2_MODULE = -8077
    NO_FP_OPTION = -8078
    FILE_DOES_NOT_EXIST = -8079
    SD_WARNING_DAQ_POINTS_ODD_NUM = -9000

    @classmethod
    def getErrorMessage(cls, errorNumber) :
        cls._SD_Object__core_dll.SD_GetErrorMessage.restype = c_char_p;
        return cls._SD_Object__core_dll.SD_GetErrorMessage(errorNumber).decode();

class SD_Object_Type :
    HVI = 1;
    AOU = 2;
    TDC = 3;
    DIO = 4;
    WAVE = 5;
    AIN = 6;
    AIO = 7;

class SD_Waveshapes :
    AOU_HIZ = -1;
    AOU_OFF = 0;
    AOU_SINUSOIDAL = 1;
    AOU_TRIANGULAR = 2;
    AOU_SQUARE = 4;
    AOU_DC = 5;
    AOU_AWG = 6;
    AOU_PARTNER = 8;

class SD_DigitalFilterModes :
    AOU_FILTER_OFF = 0;
    AOU_FILTER_FLATNESS = 1;
    AOU_FILTER_FIFTEEN_TAP = 3;

class SD_WaveformTypes :
    WAVE_ANALOG = 0;
    WAVE_IQ = 2;
    WAVE_IQPOLAR = 3;
    WAVE_DIGITAL = 5;
    WAVE_ANALOG_DUAL = 7;

class SD_ModulationTypes :
    AOU_MOD_OFF = 0;
    AOU_MOD_FM = 1;
    AOU_MOD_PHASE = 2;

    AOU_MOD_AM = 1;
    AOU_MOD_OFFSET = 2;

class SD_TriggerDirections :
    AOU_TRG_OUT = 0;
    AOU_TRG_IN = 1;

class SD_TriggerBehaviors :
    TRIGGER_NONE = 0;
    TRIGGER_HIGH = 1;
    TRIGGER_LOW = 2;
    TRIGGER_RISE = 3;
    TRIGGER_FALL = 4;

class SD_MarkerModes :
    DISABLED = 0;
    START = 1;
    START_AFTER_DELAY = 2;
    EVERY_CYCLE = 3;

class SD_TriggerValue :
    LOW = 0;
    HIGH = 1;

class SD_TriggerPolarity :
    ACTIVE_LOW  = 0;
    ACTIVE_HIGH  = 1;

class SD_SyncModes :
    SYNC_NONE = 0;
    SYNC_CLK10 = 1;

class SD_QueueMode :
    ONE_SHOT = 0;
    CYCLIC = 1;

class SD_ResetMode :
    LOW = 0;
    HIGH = 1;
    PULSE = 2;

class SD_AddressingMode :
    AUTOINCREMENT  = 0;
    FIXED = 1;

class SD_AccessMode :
    NONDMA = 0;
    DMA = 1;
    
class SD_FpgaTriggerDirection :
    IN = 0;
    INOUT = 1;

class SD_TriggerModes :
    AUTOTRIG = 0;
    VIHVITRIG = 1;
    SWHVITRIG = 1;
    EXTTRIG = 2;
    HWDIGTRIG = 2;
    HWANATRIG = 3;
    SWHVITRIG_CYCLE = 5;
    EXTTRIG_CYCLE = 6;
    ANALOGAUTOTRIG = 11;

class SD_TriggerExternalSources :
    TRIGGER_EXTERN = 0;
    TRIGGER_PXI = 4000;
    TRIGGER_PXI0 = 4000;
    TRIGGER_PXI1 = 4001;
    TRIGGER_PXI2 = 4002;
    TRIGGER_PXI3 = 4003;
    TRIGGER_PXI4 = 4004;
    TRIGGER_PXI5 = 4005;
    TRIGGER_PXI6 = 4006;
    TRIGGER_PXI7 = 4007;

class SD_IOdirections :
    DIR_IN = 0;
    DIR_OUT = 1;

class SD_PinDirections :
    DIR_IN = 0;
    DIR_OUT = 1;

class SD_Strobe :
    STROBE_OFF = 0;
    STROBE_ON = 1;

    STROBE_LEVEL = 2;##0b10;
    STROBE_EDGERISE = 1;
    STROBE_EDGEFALL = 0;

class SD_DebouncingTypes :
    DEBOUNCING_NONE = 0;
    DEBOUNCING_LOW = 2;##0b10;
    DEBOUNCING_HIGH = 3;##0b11;


class SD_Compatibility :
    LEGACY = 0;
    KEYSIGHT = 1;

class SD_Wave(SD_Object) :
    PADDING_ZERO = 0;
    PADDING_REPEAT = 1;

    def newFromFile(self, waveformFile) :
        self._SD_Object__handle = self._SD_Object__core_dll.SD_Wave_newFromFile(waveformFile.encode());

        return self._SD_Object__handle;

    def __del__(self):
        self._SD_Object__core_dll.SD_Wave_delete(self._SD_Object__handle)
    
    def newFromArrayDoubleNP(self, waveformType, waveformDataA, waveformDataB = None):
        if len(waveformDataA) > 0 and (waveformDataB is None or len(waveformDataA) == len(waveformDataB)):
            dataA_np = to_numpy_float(waveformDataA)
            waveform_dataA_C = dataA_np.ctypes.data_as(POINTER(c_double*len(dataA_np))).contents
            if waveformDataB is None:
                waveform_dataB_C = c_void_p(0)
            else:
                dataB_np = to_numpy_float(waveformDataB)
                waveform_dataB_C = dataB_np.ctypes.data_as(POINTER(c_double*len(dataB_np))).contents
            
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Wave_newFromArrayDouble(waveformType, waveform_dataA_C._length_, waveform_dataA_C, waveform_dataB_C)
            
            return self._SD_Object__handle
        else :
            self._SD_Object__handle = 0

        return SD_Error.INVALID_VALUE

    def newFromArrayDouble(self, waveformType, waveformDataA, waveformDataB = None) :
        if len(waveformDataA) > 0 and (waveformDataB is None or len(waveformDataA) == len(waveformDataB)) :
            waveform_dataA_C = (c_double * len(waveformDataA))(*waveformDataA);
            if waveformDataB is None:
                waveform_dataB_C = c_void_p(0);
            else :
                waveform_dataB_C = (c_double * len(waveformDataB))(*waveformDataB);

            self._SD_Object__handle = self._SD_Object__core_dll.SD_Wave_newFromArrayDouble(waveformType, waveform_dataA_C._length_, waveform_dataA_C, waveform_dataB_C);

            return self._SD_Object__handle;
        else :
            self._SD_Object__handle = 0;

            return SD_Error.INVALID_VALUE;

    def newFromArrayInteger(self, waveformType, waveformDataA, waveformDataB = None) :
        if len(waveformDataA) > 0 and (waveformDataB is None or len(waveformDataA) == len(waveformDataB)) :
            waveform_dataA_C = (c_int * len(waveformDataA))(*waveformDataA);
            if waveformDataB is None:
                waveform_dataB_C = c_void_p(0);
            else :
                waveform_dataB_C = (c_int * len(waveformDataB))(*waveformDataB);

            self._SD_Object__handle = self._SD_Object__core_dll.SD_Wave_newFromArrayInteger(waveformType, waveform_dataA_C._length_, waveform_dataA_C, waveform_dataB_C);

            return self._SD_Object__handle;
        else :
            self._SD_Object__handle = 0;

    def getStatus(self) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Wave_getStatus(self._SD_Object__handle);
        else :
            return SD_Error.CREATING_WAVE;

    def getType(self) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Wave_getType(self._SD_Object__handle);
        else :
            return SD_Error.CREATING_WAVE;


class SD_SandBoxRegister(SD_Object):
   
    def __init__(self, moduleId, registerId):
        self._SD_Object__handle = moduleId
        self._SD_Register_Id = registerId
        [self.Address, self.Length, self.AccessType, self.Name] =self.__getRegisterInfo()
    
    def __getRegisterInfo(self) :
        if self._SD_Object__handle > 0 :
            address = c_int(0)
            length = c_int(0)
            name = ''.rjust(100, '\0').encode();
            accessType = ''.rjust(20, '\0').encode();
            error =  self._SD_Object__core_dll.SD_Module_FPGAgetRegisterInfo(self._SD_Object__handle, self._SD_Register_Id, byref(address),byref(length),accessType, name);
                
            if error < 0 :
                return error
            else :
                return [address.value,length.value, accessType.decode(), name.decode()]
        else :
            return SD_Error.MODULE_NOT_OPENED;
    
    def readRegisterBuffer(self, indexOffset, bufferSize, addressMode, accessMode) :
        if self._SD_Object__handle > 0 :
            if bufferSize > 0 :
                bufferSize = int(bufferSize)
                data = (c_int * bufferSize)()
                error = self._SD_Object__core_dll.SD_Module_FPGAreadRegisterBuffer(self._SD_Object__handle, self._SD_Register_Id,indexOffset, data, bufferSize, addressMode, accessMode)

                if error < 0 :
                    return error
                else :
                    return np.array(cast(data, POINTER(c_int*bufferSize)).contents)
            else :
                return SD_Error.INVALID_VALUE
        else :
            return SD_Error.MODULE_NOT_OPENED

    def writeRegisterBuffer(self, indexOffset, buffer, addressMode, accessMode) :
        if self._SD_Object__handle > 0 :
            if len(buffer) > 0 :
                data = (c_int * len(buffer))(*buffer);
                return self._SD_Object__core_dll.SD_Module_FPGAwriteRegisterBuffer(self._SD_Object__handle, self._SD_Register_Id, indexOffset, data, data._length_, addressMode, accessMode);
            else :
                return SD_Error.INVALID_VALUE;
        else :
            return SD_Error.MODULE_NOT_OPENED;
        
    def writeRegisterInt32(self, data) :
        if self._SD_Object__handle > 0 :
                return self._SD_Object__core_dll.SD_Module_FPGAwriteRegisterInt32(self._SD_Object__handle, self._SD_Register_Id, data);
        else :
            return SD_Error.MODULE_NOT_OPENED;
 
    def readRegisterInt32(self) :
        if self._SD_Object__handle > 0 :
                data = c_int(0)
                error =  self._SD_Object__core_dll.SD_Module_FPGAreadRegisterInt32(self._SD_Object__handle, self._SD_Register_Id, byref(data));
                
                if error < 0 :
                    return error
                else :
                    return data.value
        else :
            return SD_Error.MODULE_NOT_OPENED;
              
    
class SD_Module(SD_Object) :
    def openWithSerialNumber(self, partNumber, serialNumber) :
        if self._SD_Object__handle <= 0 :
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSerialNumber(partNumber.encode(), serialNumber.encode())

            if self._SD_Object__handle >= 0 and self.isHvi2Module():
                self.createHvi()
        return self._SD_Object__handle

    def openWithSerialNumberCompatibility(self, partNumber, serialNumber, compatibility) :
        if self._SD_Object__handle <= 0 :
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSerialNumberCompatibility(partNumber.encode(), serialNumber.encode(), compatibility);

            if self._SD_Object__handle >= 0 and self.isHvi2Module():
                self.createHvi()
        return self._SD_Object__handle

    def openWithSlot(self, partNumber, nChassis, nSlot, hvi=True) :
        if self._SD_Object__handle <= 0 :
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSlot(partNumber.encode(), nChassis, nSlot)

            if hvi:
                if self._SD_Object__handle >= 0 and self.isHvi2Module():
                    self.createHvi()
        return self._SD_Object__handle

    def openWithOptions(self, partNumber, nChassis, nSlot, options) :
        if self._SD_Object__handle <= 0 :
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithOptions(partNumber.encode(), nChassis, nSlot, options.encode())

            if self._SD_Object__handle >= 0 and self.isHvi2Module():
                self.createHvi()
        return self._SD_Object__handle

    def openWithSlotCompatibility(self, partNumber, nChassis, nSlot, compatibility):
        if self._SD_Object__handle <= 0:
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSlotCompatibility(partNumber.encode(), nChassis, nSlot, compatibility);

            if self._SD_Object__handle >= 0 and self.isHvi2Module():
                self.createHvi()
        return self._SD_Object__handle

    def close(self) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_close(self._SD_Object__handle)

        return self._SD_Object__handle

    def isOpen(self) :
        return self._SD_Object__handle > 0

    def getType(self) :
        objectType = self._SD_Object__core_dll.SD_Module_getType(self._SD_Object__handle)

        if objectType < SD_Object_Type.AOU or objectType > SD_Object_Type.AIO or objectType == SD_Object_Type.WAVE:
            objectType = SD_Error.INVALID_MODULEID

        return objectType

    @classmethod
    def moduleCount(cls):
        return cls._SD_Object__core_dll.SD_Module_count();

    @classmethod
    def getProductNameBySlot(cls, chassis, slot) :
        buffer = ''.rjust(50, '\0').encode();

        error = cls._SD_Object__core_dll.SD_Module_getProductNameBySlot(chassis, slot, buffer);

        if error < 0 :
            return error;
        else :
            return cls._SD_Object__formatString(buffer);

    @classmethod
    def getProductNameByIndex(cls, index) :
        buffer = ''.rjust(50, '\0').encode();

        error = cls._SD_Object__core_dll.SD_Module_getProductNameByIndex(index, buffer);

        if error < 0 :
            return error;
        else :
            return cls._SD_Object__formatString(buffer);

    @classmethod
    def getSerialNumberBySlot(cls, chassis, slot) :
        buffer = ''.rjust(50, '\0').encode();

        error = cls._SD_Object__core_dll.SD_Module_getSerialNumberBySlot(chassis, slot, buffer);

        if error < 0 :
            return error;
        else :
            return cls._SD_Object__formatString(buffer);

    @classmethod
    def getSerialNumberByIndex(cls, index) :
        buffer = ''.rjust(50, '\0').encode();

        error = cls._SD_Object__core_dll.SD_Module_getSerialNumberByIndex(index, buffer);

        if error < 0 :
            return error;
        else :
            return cls._SD_Object__formatString(buffer);

    @classmethod
    def getTypeBySlot(cls, chassis, slot) :
        return cls._SD_Object__core_dll.SD_Module_getTypeBySlot(chassis, slot);

    @classmethod
    def getTypeByIndex(cls, index) :
        return cls._SD_Object__core_dll.SD_Module_getTypeByIndex(index);

    @classmethod
    def getChassisByIndex(cls, index) :
        return cls._SD_Object__core_dll.SD_Module_getChassisByIndex(index);

    @classmethod
    def getSlotByIndex(cls, index) :
        return cls._SD_Object__core_dll.SD_Module_getSlotByIndex(index);


    def runSelfTest(self) :
        result = 0;

        if self._SD_Object__handle > 0 :
            result = self._SD_Object__core_dll.SD_Module_runSelfTest(self._SD_Object__handle);
        else :
            result = SD_Error.MODULE_NOT_OPENED;

        return result;

    def getSerialNumber(self) :
        serial = ''.rjust(50, '\0').encode();

        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_Module_getSerialNumber.restype = c_char_p;
            return self._SD_Object__core_dll.SD_Module_getSerialNumber(self._SD_Object__handle, serial).decode();
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def getProductName(self) :
        product = ''.rjust(50, '\0').encode();

        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_Module_getProductName.restype = c_char_p;
            return self._SD_Object__core_dll.SD_Module_getProductName(self._SD_Object__handle, product).decode();
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def getFirmwareVersion(self) :
        version = ''.rjust(9, '\0').encode();

        if self._SD_Object__handle > 0 :
            retValue = self._SD_Object__core_dll.SD_Module_getFirmwareVersion(self._SD_Object__handle, version);
            if retValue >= 0 :
                return version.decode();
            else :
                return retValue;
        else :
            return SD_Error.MODULE_NOT_OPENED;
        

    def getHardwareVersion(self) :
        version = ''.rjust(9, '\0').encode();

        if self._SD_Object__handle > 0 :
            retValue = self._SD_Object__core_dll.SD_Module_getHardwareVersion(self._SD_Object__handle, version);
            if retValue >= 0 :
                return version.decode();
            else :
                return retValue;
        else :
            return SD_Error.MODULE_NOT_OPENED;
        

    def getChassis(self) :
        result = 0;

        if self._SD_Object__handle > 0 :
            result = self._SD_Object__core_dll.SD_Module_getChassis(self._SD_Object__handle);
        else :
            result = SD_Error.MODULE_NOT_OPENED;

        return result;

    def getSlot(self) :
        result = 0;

        if self._SD_Object__handle > 0 :
            result = self._SD_Object__core_dll.SD_Module_getSlot(self._SD_Object__handle);
        else :
            result = SD_Error.MODULE_NOT_OPENED;

        return result;

    def getTemperature(self) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_Module_getTemperature.restype = c_double;
            result = self._SD_Object__core_dll.SD_Module_getTemperature(self._SD_Object__handle);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def getOptions(self, optionkey) :
        varToFill = ''.rjust(200, '\0').encode();
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_Module_getOptions.restype = c_char_p;
            result = self._SD_Object__core_dll.SD_Module_getOptions(self._SD_Object__handle,optionkey.encode(), varToFill, len(varToFill), self.getType()).decode();
            return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;


    ## PXItrigger
    def PXItriggerWrite(self, trigger, value) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_PXItriggerWrite(self._SD_Object__handle, trigger, value);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def PXItriggerRead(self, trigger) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_PXItriggerRead(self._SD_Object__handle, trigger);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    ## External Trigger Lines
    def translateTriggerPXItoExternalTriggerLine(self, trigger) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_translateTriggerPXItoExternalTriggerLine(self._SD_Object__handle, trigger);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def translateTriggerIOtoExternalTriggerLine(self, trigger) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_translateTriggerIOtoExternalTriggerLine(self._SD_Object__handle, trigger);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    ## FPGA
    def FPGAload(self, fileName) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_FPGAload(self._SD_Object__handle, fileName.encode());
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def FPGAreset(self, mode) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_FPGAreset(self._SD_Object__handle, mode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def FPGAgetSandBoxRegister(self, registerName):
        if self._SD_Object__handle > 0 :
                id =  self._SD_Object__core_dll.SD_Module_FPGAgetRegisterId(self._SD_Object__handle, registerName.encode());
                
                if id < 0 :
                    return id
                else :
                    return SD_SandBoxRegister(self._SD_Object__handle,id )
        else :
            return SD_Error.MODULE_NOT_OPENED;
    
    def FPGAgetSandBoxRegisters(self, count):
        if self._SD_Object__handle > 0 :
            count = int(count)
            data = (c_int * count)()
            error =  self._SD_Object__core_dll.SD_Module_FPGAgetRegisterIds(self._SD_Object__handle, data, count);
                
            if error < 0 :
                return error
            else :
                registers = []
                    
                for id in np.array(cast(data, POINTER(c_int*count)).contents):
                    id = int(id)
                    registers.append(SD_SandBoxRegister(self._SD_Object__handle,id ))
                
                return registers
            
        else :
            return SD_Error.MODULE_NOT_OPENED;
       
    def FPGATriggerConfig(self, externalSource, direction, polarity, syncMode, delay5Tclk) :
        if self._SD_Object__handle > 0 :
                return self._SD_Object__core_dll.SD_Module_FPGATriggerConfig(self._SD_Object__handle, externalSource, direction, polarity, syncMode, delay5Tclk);
        else :
            return SD_Error.MODULE_NOT_OPENED;
        

    # HVI2
    def isHvi2Module(self):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_Module_isHvi2Module(self._SD_Object__handle)
        else :
            return SD_Error.MODULE_NOT_OPENED

    @abstractmethod
    def createHvi(self):
        pass

    def getHviEngineUid(self, engineId):
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_Module_getHviEngineUid.restype = c_longlong;
            return self._SD_Object__core_dll.SD_Module_getHviEngineUid(self._SD_Object__handle, engineId)
        else :
            return SD_Error.MODULE_NOT_OPENED

class Engine:
    __kMasterEngineId = 0

    def __init__(self, module):
        self.__module = module

    @property
    def main_engine(self):
        return self.__module.getHviEngineUid(self.__kMasterEngineId)


class TriggerModule:
    @property
    def pxi_0(self):
        return 0
    @property
    def pxi_1(self):
        return 1
    @property
    def pxi_2(self):
        return 2
    @property
    def pxi_3(self):
        return 3
    @property
    def pxi_4(self):
        return 4
    @property
    def pxi_5(self):
        return 5
    @property
    def pxi_6(self):
        return 6
    @property
    def pxi_7(self):
        return 7
    @property
    def front_panel_1(self):
        return 8


class TriggerAIO(TriggerModule):
    @property
    def front_panel_2(self):
        return 9    


class ActionAwg:

    def __init__(self, module):
        self.__module = module

    @property
    def ch1_reset_phase(self):
        return self.__module.getAction(constants.SD_AOU_Action_CH1ResetPhase)

    @property
    def ch2_reset_phase(self):
        return self.__module.getAction(constants.SD_AOU_Action_CH2ResetPhase)

    @property
    def ch3_reset_phase(self):
        return self.__module.getAction(constants.SD_AOU_Action_CH3ResetPhase)

    @property
    def ch4_reset_phase(self):
        return self.__module.getAction(constants.SD_AOU_Action_CH4ResetPhase)

    @property
    def awg1_start(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1Start)

    @property
    def awg2_start(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2Start)

    @property
    def awg3_start(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3Start)

    @property
    def awg4_start(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4Start)

    @property
    def awg1_stop(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1Stop)

    @property
    def awg2_stop(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2Stop)

    @property
    def awg3_stop(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3Stop)

    @property
    def awg4_stop(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4Stop)

    @property
    def awg1_pause(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1Pause)

    @property
    def awg2_pause(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2Pause)

    @property
    def awg3_pause(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3Pause)

    @property
    def awg4_pause(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4Pause)

    @property
    def awg1_resume(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1Resume)

    @property
    def awg2_resume(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2Resume)

    @property
    def awg3_resume(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3Resume)

    @property
    def awg4_resume(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4Resume)

    @property
    def awg1_trigger(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1Trigger)

    @property
    def awg2_trigger(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2Trigger)

    @property
    def awg3_trigger(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3Trigger)

    @property
    def awg4_trigger(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4Trigger)

    @property
    def awg1_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1JumpNextWaveform)

    @property
    def awg2_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2JumpNextWaveform)

    @property
    def awg3_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3JumpNextWaveform)

    @property
    def awg4_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4JumpNextWaveform)

    @property
    def awg1_queue_flush(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG1QueueFlush)

    @property
    def awg2_queue_flush(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG2QueueFlush)

    @property
    def awg3_queue_flush(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG3QueueFlush)

    @property
    def awg4_queue_flush(self):
        return self.__module.getAction(constants.SD_AOU_Action_AWG4QueueFlush)

    @property
    def fpga_user_0(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga0)

    @property
    def fpga_user_1(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga1)

    @property
    def fpga_user_2(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga2)

    @property
    def fpga_user_3(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga3)

    @property
    def fpga_user_4(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga4)

    @property
    def fpga_user_5(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga5)

    @property
    def fpga_user_6(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga6)

    @property
    def fpga_user_7(self):
        return self.__module.getAction(constants.SD_AOU_Action_UserFpga7)


class Event:

    def __init__(self, module):
        self.__module = module

    @property
    def awg1_queue_empty(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1QueueEmpty)

    @property
    def awg2_queue_empty(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2QueueEmpty)

    @property
    def awg3_queue_empty(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3QueueEmpty)

    @property
    def awg4_queue_empty(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4QueueEmpty)

    @property
    def awg1_queue_full(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1QueueFull)

    @property
    def awg2_queue_full(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2QueueFull)

    @property
    def awg3_queue_full(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3QueueFull)

    @property
    def awg4_queue_full(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4QueueFull)

    @property
    def awg1_underrun(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1Underrun)

    @property
    def awg2_underrun(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2Underrun)

    @property
    def awg3_underrun(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3Underrun)

    @property
    def awg4_underrun(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4Underrun)

    @property
    def awg1_queue_end(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1QueueEnd)

    @property
    def awg2_queue_end(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2QueueEnd)

    @property
    def awg3_queue_end(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3QueueEnd)

    @property
    def awg4_queue_end(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4QueueEnd)

    @property
    def awg1_waveform_start(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1WfStart)

    @property
    def awg2_waveform_start(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2WfStart)

    @property
    def awg3_waveform_start(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3WfStart)

    @property
    def awg4_waveform_start(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4WfStart)

    @property
    def awg1_queue_marker(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1QueueMarker)

    @property
    def awg2_queue_marker(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2QueueMarker)

    @property
    def awg3_queue_marker(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3QueueMarker)

    @property
    def awg4_queue_marker(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4QueueMarker)

    @property
    def awg1_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1QueueFlushed)

    @property
    def awg2_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2QueueFlushed)

    @property
    def awg3_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3QueueFlushed)

    @property
    def awg4_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4QueueFlushed)

    @property
    def awg1_queue_running(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1QueueRunning)

    @property
    def awg2_queue_running(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2QueueRunning)

    @property
    def awg3_queue_running(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3QueueRunning)

    @property
    def awg4_queue_running(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4QueueRunning)

    @property
    def fpga_user_0(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback0)

    @property
    def fpga_user_1(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback1)

    @property
    def fpga_user_2(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback2)

    @property
    def fpga_user_3(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback3)

    @property
    def fpga_user_4(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback4)

    @property
    def fpga_user_5(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback5)

    @property
    def fpga_user_6(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback6)

    @property
    def fpga_user_7(self):
        return self.__module.getEvent(constants.SD_AOU_Event_UserFpgaLoopback7)
    
    @property
    def awg1_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG1TriggerLoopback)
 
    @property
    def awg2_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG2TriggerLoopback)
    
    @property
    def awg3_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG3TriggerLoopback)
    
    @property
    def awg4_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AOU_Event_AWG4TriggerLoopback)


class InstructionParameter:
    def __init__(self, module, attributeId):
        self.__module = module
        self.__attributeId = attributeId

    @property
    def id(self):
        return self.__module.getAttributeId64(self.__attributeId)



class SetAmplitudeInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetAmplitude_Parameters_Channel_Id)
        self.__value = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetAmplitude_Parameters_Value_Id)

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetAmplitude_Id)
    
    @property
    def channel(self):
        return self.__channel
    
    @property
    def value(self):
         return self.__value

    
class WaveShapeValue:
    def __init__(self, module):
        self.__module = module

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_Id)

    @property
    def HIZ(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_HIZ_Id)

    @property
    def AOU_OFF(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_OFF_Id)

    @property
    def AOU_SINUSOIDAL(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_SINUSOIDAL_Id)

    @property
    def AOU_TRIANGULAR(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_TRIANGULAR_Id)

    @property
    def AOU_SQUARE(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_SQUARE_Id)

    @property
    def AOU_DC(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_DC_Id)

    @property
    def AOU_AWG(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_AWG_Id)

    @property
    def AOU_PARTNER_CHANNEL(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Value_AOU_PARTNER_Id)
  
    
    
class SetWaveshapeInstruction:
    def __init__(self,module):
        self.__module = module
        self.__value = WaveShapeValue(module)
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetWaveshape_Parameters_Channel_Id)

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetWaveshape_Id)
    
    @property
    def channel(self):
        return self.__channel

    @property
    def value(self):
         return self.__value



class SetOffsetInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetOffset_Parameters_Channel_Id)
        self.__value = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetOffset_Parameters_Value_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetOffset_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def value(self):
         return self.__value

 
  
class SetFrequencyInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetFrequency_Parameters_Channel_Id)
        self.__value = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetFrequency_Parameters_Value_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetFrequency_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def value(self):
         return self.__value

   
  
class SetPhaseInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetPhase_Parameters_Channel_Id)
        self.__value = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_SetPhase_Parameters_Value_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_SetPhase_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def value(self):
         return self.__value
    
    
class ModeType:
    def __init__(self, module):
        self.__module = module

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_Id)

    @property
    def AOU_MOD_OFF(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_AOU_MOD_OFF_Id)

    @property
    def AOU_MOD_FM(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_AOU_MOD_FM_Id)

    @property
    def AOU_MOD_PM(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_AOU_MOD_PHASE_Id)



class ModulationFreqPhaseConfigInstruction:
    def __init__(self,module):
        self.__module = module
        self.__modetype = ModeType(module)
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_Channel_Id)
        self.__devgain = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_DevGain_Id )


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def modulation_type(self):
         return self.__modetype
     
    @property
    def deviation_gain(self):
         return self.__devgain


class ModulationAmpConfigModeType:
    def __init__(self, module):
        self.__module = module

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationAmpOffsetConfig_Parameters_ModType_Id)

    @property
    def AOU_MOD_OFF(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_AOU_MOD_OFF_Id)

    @property
    def AOU_MOD_AM(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_AOU_MOD_AM_Id)

    @property
    def AOU_MOD_OFFSET(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationFreqPhaseConfig_Parameters_ModType_AOU_MOD_OFFSET_Id)

 
class ModulationAmpOffsetConfigInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_ModulationAmpOffsetConfig_Parameters_Channel_Id)
        self.__modetype = ModulationAmpConfigModeType(module)
        self.__gain = InstructionParameter(module, constants.SD_AOU_Hvi_Instructions_ModulationAmpOffsetConfig_Parameters_Gain_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_ModulationAmpOffsetConfig_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def modulation_type(self):
        return self.__modetype

    @property
    def deviation_gain(self):
        return self.__gain
    


class TriggerMode:
    def __init__(self, module):
        self.__module = module

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_TriggerMode_Id)

    @property
    def AUTOTRIG(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_TriggerMode_AUTOTRIG_Id)

    @property
    def SWHVITRIG(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_TriggerMode_SWHVITRIG_Id)

    @property
    def EXTTRIG(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_TriggerMode_EXTTRIG_Id)


    @property
    def SWHVITRIG_CYCLE(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_TriggerMode_SWHVITRIG_CYCLE_Id)

    @property
    def EXTTRIG_CYCLE(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_TriggerMode_EXTTRIG_CYCLE_Id)



class QueueWaveformInstruction:
    def __init__(self,module):
        self.__module = module
        self.__triggerMode = TriggerMode(module)
        self.__channel = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_Channel_Id)
        self.__waveform = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_WaveformId_Id)
        self.__cycles = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_Cycles_Id)
        self.__startdelay = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_StartDelay_Id)
        self.__prescaler = InstructionParameter(module,constants.SD_AOU_Hvi_Instructions_QueueWaveform_Parameters_Prescaler_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AOU_Hvi_Instructions_QueueWaveform_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def waveform_number(self):
        return self.__waveform

    @property
    def cycles(self):
        return self.__cycles

    @property
    def start_delay(self):
        return self.__startdelay

    @property
    def prescaler(self):
        return self.__prescaler

    @property
    def trigger_mode(self):
         return self.__triggerMode
    

class InstructionAWG:
    def __init__(self, module):
        self.__module = module
        self.__setamplitude = SetAmplitudeInstruction(module)
        self.__setwaveshape = SetWaveshapeInstruction(module)
        self.__setoffset = SetOffsetInstruction(module)
        self.__setfrequency = SetFrequencyInstruction(module)
        self.__setphase = SetPhaseInstruction(module)
        self.__modulationfreqphaseconfig = ModulationFreqPhaseConfigInstruction(module)
        self.__modulationampoffsetconfig = ModulationAmpOffsetConfigInstruction(module)
        self.__queuewaveform = QueueWaveformInstruction(module)
        
    
    @property
    def set_amplitude(self):
        return self.__setamplitude
    
    @property
    def set_waveshape(self):
        return self.__setwaveshape
    
    @property
    def set_offset(self):
        return self.__setoffset
    
    @property
    def set_frequency(self):
        return self.__setfrequency
    
    @property
    def set_phase(self):
        return self.__setphase
    
    @property
    def modulation_angle_config(self):
        return self.__modulationfreqphaseconfig
    
    @property
    def modulation_amplitude_config(self):
        return self.__modulationampoffsetconfig
 
    @property
    def queue_waveform(self):
        return self.__queuewaveform


class SD_AOUHvi:
    def __init__(self, module):
        self.__module = module
        self.__engines = Engine(module)
        self.__triggers = TriggerModule()
        self.__actions = ActionAwg(module)
        self.__events = Event(module)
        self.__instructions = InstructionAWG(module)

    @property
    def engines(self):
        return self.__engines

    @property
    def triggers(self):
        return self.__triggers

    @property
    def actions(self):
        return self.__actions

    @property
    def events(self):
        return self.__events

    @property
    def instruction_set(self):
        return self.__instructions


class SD_AOU(SD_Module):

    def __init__(self):
        super(SD_AOU, self).__init__()
        self.__hvi = None

    def createHvi(self):
        self.__hvi = SD_AOUHvi(self)

    @property
    def hvi(self):
        return self.__hvi

    def AWGqueueIsFull(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueIsFull(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;
        
    def AWGqueueIsEmpty(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueIsEmpty(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;
        
    def AWGqueueRemaining(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueRemaining(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockGetFrequency(self) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AOU_clockGetFrequency.restype = c_double;
            result = self._SD_Object__core_dll.SD_AOU_clockGetFrequency(self._SD_Object__handle);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockGetSyncFrequency(self) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AOU_clockGetSyncFrequency.restype = c_double;
            result = self._SD_Object__core_dll.SD_AOU_clockGetSyncFrequency(self._SD_Object__handle);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockSetFrequency(self, frequency, mode = 1) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AOU_clockSetFrequency.restype = c_double;
            result = self._SD_Object__core_dll.SD_AOU_clockSetFrequency(self._SD_Object__handle, c_double(frequency), mode);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockResetPhase(self, triggerBehavior, triggerSource, skew = 0.0):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_clockResetPhase(self._SD_Object__handle, triggerBehavior, triggerSource, c_double(skew));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def setDigitalFilterMode(self, mode):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_setDigitalFilterMode(self._SD_Object__handle, mode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelAmplitude(self, nChannel, amplitude):
        if self._SD_Object__handle > 0:
            return self._SD_Object__core_dll.SD_AOU_channelAmplitude(self._SD_Object__handle, nChannel, c_double(amplitude))
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelOffset(self, nChannel, offset) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_channelOffset(self._SD_Object__handle, nChannel, c_double(offset));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelWaveShape(self, nChannel, waveShape) :
        if self._SD_Object__handle > 0:
            return self._SD_Object__core_dll.SD_AOU_channelWaveShape(self._SD_Object__handle, nChannel, waveShape);
        else :
            return SD_Error.MODULE_NOT_OPENED

    def channelFrequency(self, nChannel, frequency) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AOU_channelFrequency.restype = c_double;
            return self._SD_Object__core_dll.SD_AOU_channelFrequency(self._SD_Object__handle, nChannel, c_double(frequency));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelPhase(self, nChannel, phase) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_channelPhase(self._SD_Object__handle, nChannel, c_double(phase));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelPhaseReset(self, nChannel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_channelPhaseReset(self._SD_Object__handle, nChannel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelPhaseResetMultiple(self, channelMask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_channelPhaseResetMultiple(self._SD_Object__handle, channelMask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def modulationAngleConfig(self, nChannel, modulationType, deviationGain) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_modulationAngleConfig(self._SD_Object__handle, nChannel, modulationType, c_double(deviationGain));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def modulationAmplitudeConfig(self, nChannel, modulationType, deviationGain) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_modulationAmplitudeConfig(self._SD_Object__handle, nChannel, modulationType, c_double(deviationGain));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def modulationIQconfig(self, nChannel, enable) :
        if self._SD_Object__handle > 0:
            return self._SD_Object__core_dll.SD_AOU_modulationIQconfig(self._SD_Object__handle, nChannel, enable);
        else :
            return SD_Error.MODULE_NOT_OPENED;
    def clockIOconfig(self, clockConfig) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_clockIOconfig(self._SD_Object__handle, clockConfig);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def triggerIOconfig(self, direction) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_triggerIOconfig(self._SD_Object__handle, direction);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def triggerIOwrite(self, value, syncMode = 1) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_triggerIOwrite(self._SD_Object__handle, value, syncMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def triggerIOread(self) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_triggerIOread(self._SD_Object__handle);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformReLoad(self, waveformObject, waveformNumber, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_waveformReLoad(self._SD_Object__handle, waveformObject._SD_Object__handle, waveformNumber, paddingMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformReLoadArrayInt16(self, waveformType, dataRaw, waveformNumber, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            if len(dataRaw) > 0 :
                dataC = (c_short * len(dataRaw))(*dataRaw);
                return self._SD_Object__core_dll.SD_AOU_waveformReLoadArrayInt16(self._SD_Object__handle, waveformType, dataC._length_, dataC, waveformNumber, paddingMode);
            else :
                return SD_Error.INVALID_VALUE;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformLoad(self, waveformObject, waveformNumber, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_waveformLoad(self._SD_Object__handle, waveformObject._SD_Object__handle, waveformNumber, paddingMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformLoadInt16(self, waveformType, dataRaw, waveformNumber, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            if len(dataRaw) > 0 :
                dataC = (c_short * len(dataRaw))(*dataRaw);
                return self._SD_Object__core_dll.SD_AOU_waveformLoadArrayInt16(self._SD_Object__handle, waveformType, dataC._length_, dataC, waveformNumber, paddingMode);
            else :
                return SD_Error.INVALID_VALUE;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformAddToList(self, waveformObject, waveformNumber, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_waveformAddToList(self._SD_Object__handle, waveformObject._SD_Object__handle, waveformNumber, paddingMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformListLoad(self) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_waveformListLoad(self._SD_Object__handle);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def waveformFlush(self) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_waveformFlush(self._SD_Object__handle);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGqueueWaveform(self, nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueWaveform(self._SD_Object__handle, nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGstartMultiple(self, AWGmask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGstartMultiple(self._SD_Object__handle, AWGmask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGstopMultiple(self, AWGmask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGstopMultiple(self._SD_Object__handle, AWGmask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGresumeMultiple(self, AWGmask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGresumeMultiple(self._SD_Object__handle, AWGmask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGpauseMultiple(self, AWGmask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGpauseMultiple(self._SD_Object__handle, AWGmask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGtriggerMultiple(self, AWGmask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGtriggerMultiple(self._SD_Object__handle, AWGmask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGjumpNextWaveformMultiple(self, AWGmask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGjumpNextWaveformMultiple(self._SD_Object__handle, AWGmask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGstart(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGstart(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGstop(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGstop(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGresume(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGresume(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGpause(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGpause(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGtrigger(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGtrigger(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGjumpNextWaveform(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGjumpNextWaveform(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGflush(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGflush(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGisRunning(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGisRunning(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGnWFplaying(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGnWFplaying(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def __AWGfromArrayInt(self, nAWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveformDataA, waveformDataB, paddingMode) :
        waveform_dataA_C = (c_int * len(waveformDataA))(*waveformDataA);
        if waveformDataB is None:
            waveform_dataB_C = c_void_p(0);
        else :
            waveform_dataB_C = (c_int * len(waveformDataB))(*waveformDataB);

        return self._SD_Object__core_dll.SD_AOU_AWGfromArrayInteger(self._SD_Object__handle, nAWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveform_dataA_C._length_, waveform_dataA_C, waveform_dataB_C, paddingMode)

    def AWGfromArray(self, nAWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveformDataA, waveformDataB = None, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            if len(waveformDataA) > 0 and (waveformDataB is None or len(waveformDataA) == len(waveformDataB)) :
                if waveformType == SD_WaveformTypes.WAVE_DIGITAL :
                    return self.__AWGfromArrayInt(nAWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveformDataA,waveformDataB, paddingMode)
                else :
                    waveform_dataA_C = (c_double * len(waveformDataA))(*waveformDataA);
                    if waveformDataB is None:
                        waveform_dataB_C = c_void_p(0);
                    else :
                        waveform_dataB_C = (c_double * len(waveformDataB))(*waveformDataB);

                    return self._SD_Object__core_dll.SD_AOU_AWGfromArray(self._SD_Object__handle, nAWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveform_dataA_C._length_, waveform_dataA_C, waveform_dataB_C, paddingMode)
            else :
                return SD_Error.INVALID_VALUE
        else :
            return SD_Error.MODULE_NOT_OPENED

    def AWGFromFile(self, nAWG, waveformFile, triggerMode, startDelay, cycles, prescaler, paddingMode = 0) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGfromFile(self._SD_Object__handle, nAWG, waveformFile.encode(), triggerMode, startDelay, cycles, prescaler, paddingMode)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def AWGtriggerExternalConfig(self, nAWG, externalSource, triggerBehavior, sync = SD_SyncModes.SYNC_CLK10) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGtriggerExternalConfig(self._SD_Object__handle, nAWG, externalSource, triggerBehavior, sync);
        else :
            return SD_Error.MODULE_NOT_OPENED;


    def AWGqueueConfig(self, nAWG, mode) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueConfig(self._SD_Object__handle, nAWG, mode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGqueueConfigRead(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueConfigRead(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGfreezeOnStopEnable(self, nAWG, mode) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGfreezeOnStopEnable(self._SD_Object__handle, nAWG, mode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGisFreezeOnStopEnabled(self, nAWG) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGisFreezeOnStopEnabled(self._SD_Object__handle, nAWG);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGqueueMarkerConfig(self, nAWG, markerMode, trgPXImask, trgIOmask, value, syncMode, length, delay) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueMarkerConfig(self._SD_Object__handle, nAWG, markerMode, trgPXImask, trgIOmask, value, syncMode, length, delay);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def AWGqueueSyncMode(self, nAWG, syncMode) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_AWGqueueSyncMode(self._SD_Object__handle, nAWG, syncMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def voltsToInt(self, volts) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_voltsToInt(self._SD_Object__handle, c_double(volts));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def freqToInt(self, freq) :
        converted = c_longlong(0);
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AOU_freqToInt.restype = c_longlong;
            converted = self._SD_Object__core_dll.SD_AOU_freqToInt(self._SD_Object__handle, c_double(freq));
            return converted;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def phaseToInt(self, phase) :
        converted = c_longlong(0);
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AOU_phaseToInt.restype = c_longlong;
            converted = self._SD_Object__core_dll.SD_AOU_phaseToInt(self._SD_Object__handle, c_double(phase));
            return converted;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def freqGainToInt(self, freqGain) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_freqGainToInt(self._SD_Object__handle, c_double(freqGain));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def phaseGainToInt(self, phaseGain) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_phaseGainToInt(self._SD_Object__handle, c_double(phaseGain));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def getAction(self, actionId):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_getAction(self._SD_Object__handle, actionId)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def getEvent(self, eventId):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AOU_getEvent(self._SD_Object__handle, eventId)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def getAttributeId64(self, attributeId):
        if self._SD_Object__handle > 0 :
            id = c_longlong(0);           
            self._SD_Object__core_dll.SD_Module_Hvi_getAttributeId64(self._SD_Object__handle, attributeId, byref(id))
            return id.value
        else :
            return SD_Error.MODULE_NOT_OPENED
            


class SD_AIN_TriggerMode :
    RISING_EDGE = 1;
    FALLING_EDGE = 2;
    BOTH_EDGES = 3;


class AIN_Coupling :
    AIN_COUPLING_DC = 0;
    AIN_COUPLING_AC = 1;

class AIN_Impedance :
    AIN_IMPEDANCE_HZ = 0;
    AIN_IMPEDANCE_50 = 1;

class ActionDig:

    def __init__(self, module):
        self.__module = module

    @property
    def daq1_start(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ1Start)

    @property
    def daq2_start(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ2Start)

    @property
    def daq3_start(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ3Start)

    @property
    def daq4_start(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ4Start)

    @property
    def daq1_stop(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ1Stop)

    @property
    def daq2_stop(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ2Stop)

    @property
    def daq3_stop(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ3Stop)

    @property
    def daq4_stop(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ4Stop)

    @property
    def daq1_resume(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ1Resume)

    @property
    def daq2_resume(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ2Resume)

    @property
    def daq3_resume(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ3Resume)

    @property
    def daq4_resume(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ4Resume)

    @property
    def daq1_trigger(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ1Trigger)

    @property
    def daq2_trigger(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ2Trigger)

    @property
    def daq3_trigger(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ3Trigger)

    @property
    def daq4_trigger(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ4Trigger)

    @property
    def daq1_flush(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ1Flush)

    @property
    def daq2_flush(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ2Flush)

    @property
    def daq3_flush(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ3Flush)

    @property
    def daq4_flush(self):
        return self.__module.getAction(constants.SD_AIN_Action_DAQ4Flush)

    @property
    def fpga_user_0(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga0)

    @property
    def fpga_user_1(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga1)

    @property
    def fpga_user_2(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga2)

    @property
    def fpga_user_3(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga3)

    @property
    def fpga_user_4(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga4)

    @property
    def fpga_user_5(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga5)

    @property
    def fpga_user_6(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga6)

    @property
    def fpga_user_7(self):
        return self.__module.getAction(constants.SD_AIN_Action_UserFpga7)


class EventDig:

    def __init__(self, module):
        self.__module = module

    @property
    def daq1_empty(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ1Empty)

    @property
    def daq2_empty(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ2Empty)

    @property
    def daq3_empty(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ3Empty)

    @property
    def daq4_empty(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ4Empty)

    @property
    def daq1_running(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ1Running)

    @property
    def daq2_running(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ2Running)

    @property
    def daq3_running(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ3Running)

    @property
    def daq4_running(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ4Running)

    @property
    def daq1_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ1TriggerLoopback)

    @property
    def daq2_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ2TriggerLoopback)

    @property
    def daq3_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ3TriggerLoopback)

    @property
    def daq4_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIN_Event_DAQ4TriggerLoopback)

    @property
    def fpga_user_0(self):
        return self.__module.getEvent(constants.SD_AIN_Event_FpgaUser0)

    @property
    def fpga_user_1(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback1)

    @property
    def fpga_user_2(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback2)

    @property
    def fpga_user_3(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback3)

    @property
    def fpga_user_4(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback4)

    @property
    def fpga_user_5(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback5)

    @property
    def fpga_user_6(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback6)

    @property
    def fpga_user_7(self):
        return self.__module.getEvent(constants.SD_AIN_Event_UserFpgaLoopback7)

class TriggerModeDaqConfig:
    def __init__(self, module):
        self.__module = module


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_TriggerMode_Id)

    @property
    def AUTOTRIG(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_TriggerMode_AUTOTRIG_Id)

    @property
    def SWHVITRIG(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_TriggerMode_SWHVITRIG1_Id)

    @property
    def HWDIGTRIG(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_TriggerMode_HWDIGTRIG_Id)

    @property
    def HWANATRIG(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_TriggerMode_HWANATRIG_Id)



class DaqConfigInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_Channel_Id)
        self.__cycles = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_Cycles_Id)
        self.__pointspercycle = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_PointsPerCycle_Id)
        self.__triggermode = TriggerModeDaqConfig(module)
        self.__triggerdelay = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Parameters_TriggerDelay_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqConfigInstruction_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def cycles(self):
         return self.__cycles

    @property
    def daq_points_per_cycle(self):
         return self.__pointspercycle

    @property
    def trigger_delay(self):
         return self.__triggerdelay

    @property
    def trigger_mode(self):
         return self.__triggermode


class AnalogTrigModeChnlConfig:
    def __init__(self, module):
        self.__module = module


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Parameters_AnalogTriggerMode_Id)

    @property
    def AIN_RISING_EDGE(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Parameters_AnalogTriggerMode_AIN_RISING_EDGE_Id)

    @property
    def AIN_FALLING_EDGE(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Parameters_AnalogTriggerMode_AIN_FALLING_EDGE_Id)

    @property
    def AIN_BOTH_EDGES(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Parameters_AnalogTriggerMode_AIN_BOTH_EDGES_Id)



class ChannelTriggerConfigInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Parameters_Channel_Id)
        self.__analogtriggermode = AnalogTrigModeChnlConfig(module)
        self.__threshold = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Parameters_Threshold_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_ChannelTriggerConfigInstruction_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def analog_trigger_mode(self):
         return self.__analogtriggermode

    @property
    def threshold(self):
         return self.__threshold



class DaqAnalogTriggerConfigInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_DaqAnalogTriggerConfigInstruction_Parameters_Channel_Id)
        self.__analogtriggermask = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_DaqAnalogTriggerConfigInstruction_Parameters_AnalogTriggerMask_Id)


    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_DaqAnalogTriggerConfigInstruction_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def analog_trigger_mask(self):
         return self.__analogtriggermask



class ChannelPrescalerConfigInstruction:
    def __init__(self,module):
        self.__module = module
        self.__channel = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_ChannelPrescalerConfigInstruction_Parameters_Channel_Id)
        self.__prescaler = InstructionParameter(module,constants.SD_AIN_Hvi_Instructions_ChannelPrescalerConfigInstruction_Parameters_Prescaler_Id)

    @property
    def id(self):
        return self.__module.getAttributeId64(constants.SD_AIN_Hvi_Instructions_ChannelPrescalerConfigInstruction_Id)

    @property
    def channel(self):
        return self.__channel

    @property
    def prescaler(self):
         return self.__prescaler


class InstructionDIG:
    def __init__(self, module):
        self.__module = module
        self.__daqconfig = DaqConfigInstruction(module)
        self.__chnltriggerconfig = ChannelTriggerConfigInstruction(module)
        self.__daqanalogtriggerconfig = DaqAnalogTriggerConfigInstruction(module)
        self.__chnlprescalerconfig = ChannelPrescalerConfigInstruction(module)
    
    @property
    def daq_config(self):
        return self.__daqconfig
    
    @property
    def channel_trigger_config(self):
        return self.__chnltriggerconfig
    
    @property
    def daq_analog_trigger_config(self):
        return self.__daqanalogtriggerconfig
    
    @property
    def channel_prescaler_config(self):
        return self.__chnlprescalerconfig
    

class SD_AINHvi:

    def __init__(self, module):
        self.__module = module
        self.__engines = Engine(module)
        self.__triggers = TriggerModule()
        self.__actions = ActionDig(module)
        self.__events = EventDig(module)
        self.__instructions = InstructionDIG(module)


    @property
    def engines(self):
        return self.__engines

    @property
    def triggers(self):
        return self.__triggers

    @property
    def actions(self):
        return self.__actions

    @property
    def events(self):
        return self.__events
    
    @property
    def instruction_set(self):
        return self.__instructions


class SD_AIN(SD_Module) :

    def __init__(self):
        super(SD_AIN, self).__init__()
        self.__hvi = None

    def createHvi(self):
        self.__hvi = SD_AINHvi(self)

    @property
    def hvi(self):
        return self.__hvi
    
    def getAttributeId64(self, attributeId):
        if self._SD_Object__handle > 0 :
            id = c_longlong(0);           
            self._SD_Object__core_dll.SD_Module_Hvi_getAttributeId64(self._SD_Object__handle, attributeId, byref(id))
            return id.value
        else :
            return SD_Error.MODULE_NOT_OPENED

    def voltsToInt(self, channel, volts) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_voltsToInt(self._SD_Object__handle, channel, c_double(volts));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelInputConfig(self, channel, fullScale, impedance, coupling) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelInputConfig(self._SD_Object__handle, channel, c_double(fullScale), impedance, coupling);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelPrescalerConfig(self, channel, prescaler) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelPrescalerConfig(self._SD_Object__handle, channel, prescaler);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelPrescalerConfigMultiple(self, mask, prescaler) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelPrescalerConfigMultiple(self._SD_Object__handle, mask, prescaler);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelPrescaler(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelPrescaler(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelFullScale(self, channel) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_channelFullScale.restype = c_double;
            result = self._SD_Object__core_dll.SD_AIN_channelFullScale(self._SD_Object__handle, channel);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelMinFullScale(self, impedance, coupling) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_channelMinFullScale.restype = c_double;
            result = self._SD_Object__core_dll.SD_AIN_channelMinFullScale(self._SD_Object__handle, impedance, coupling);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelMaxFullScale(self, impedance, coupling) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_channelMaxFullScale.restype = c_double;
            result = self._SD_Object__core_dll.SD_AIN_channelMaxFullScale(self._SD_Object__handle, impedance, coupling);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelImpedance(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelImpedance(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelCoupling(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelCoupling(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def channelTriggerConfig(self, channel, analogTriggerMode, threshold) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_channelTriggerConfig(self._SD_Object__handle, channel, analogTriggerMode, c_double(threshold));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockIOconfig(self, clockConfig) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_clockIOconfig(self._SD_Object__handle, clockConfig);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockGetFrequency(self) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_clockGetFrequency.restype = c_double;
            result = self._SD_Object__core_dll.SD_AIN_clockGetFrequency(self._SD_Object__handle);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockGetSyncFrequency(self) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_clockGetSyncFrequency.restype = c_double;
            result = self._SD_Object__core_dll.SD_AIN_clockGetSyncFrequency(self._SD_Object__handle);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockSetFrequency(self, frequency, mode = 1) :
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_clockSetFrequency.restype = c_double;
            result = self._SD_Object__core_dll.SD_AIN_clockSetFrequency(self._SD_Object__handle, c_double(frequency), mode);

            if result < 0 :
                return int(result);
            else :
                return result;
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def clockResetPhase(self, triggerBehavior, PXItrigger, skew = 0.0) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_clockResetPhase(self._SD_Object__handle, triggerBehavior, PXItrigger, c_double(skew));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def triggerIOconfig(self, direction) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_triggerIOconfig(self._SD_Object__handle, direction);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def triggerIOwrite(self, value, syncMode = 1) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_triggerIOwrite(self._SD_Object__handle, value, syncMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def triggerIOread(self) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_triggerIOread(self._SD_Object__handle);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQconfig(self, channel, pointsPerCycle, nCycles, triggerDelay, triggerMode) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQconfig(self._SD_Object__handle, channel, pointsPerCycle, nCycles, triggerDelay, triggerMode);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQtriggerConfig(self, channel, digitalTriggerMode, digitalTriggerSource, analogTriggerMask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQtriggerConfig(self._SD_Object__handle, channel, digitalTriggerMode, digitalTriggerSource, analogTriggerMask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQanalogTriggerConfig(self, channel, analogTriggerMask) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQanalogTriggerConfig(self._SD_Object__handle, channel, analogTriggerMask);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQdigitalTriggerConfig(self, channel, triggerSource, triggerBehavior) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQdigitalTriggerConfig(self._SD_Object__handle, channel, triggerSource, triggerBehavior);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQtriggerExternalConfig(self, nDAQ, externalSource, triggerBehavior, sync = SD_SyncModes.SYNC_NONE) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQtriggerExternalConfig(self._SD_Object__handle, nDAQ, externalSource, triggerBehavior, sync);
        else :
            return SD_Error.MODULE_NOT_OPENED;
            
    def DAQnPoints(self,nDAQ) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQnPoints(self._SD_Object__handle, nDAQ);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQstart(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQstart(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQpause(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQpause(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQresume(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQresume(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQstop(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQstop(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQflush(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQflush(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQtrigger(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQtrigger(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQstartMultiple(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQstartMultiple(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQpauseMultiple(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQpauseMultiple(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQresumeMultiple(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQresumeMultiple(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQstopMultiple(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQstopMultiple(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQflushMultiple(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQflushMultiple(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQtriggerMultiple(self, channel) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQtriggerMultiple(self._SD_Object__handle, channel);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQread(self, nDAQ, nPoints, timeOut = 0) :
        if self._SD_Object__handle > 0 :
            if nPoints > 0 :
                data = (c_short * nPoints)()

                nPointsOrError = self._SD_Object__core_dll.SD_AIN_DAQread(self._SD_Object__handle, nDAQ, data, nPoints, timeOut)

                if nPointsOrError > 0 :
                    return np.array(cast(data, POINTER(c_short*nPointsOrError)).contents)
                elif nPointsOrError < 0 :
                    return nPointsOrError
                else :
                    return np.empty(0, dtype=np.short)
            else :
                return SD_Error.INVALID_VALUE
        else :
            return SD_Error.MODULE_NOT_OPENED

    def DAQcounterRead(self, nDAQ) :
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQcounterRead(self._SD_Object__handle, nDAQ);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQbufferPoolConfig(self, nDAQ, nPoints, timeOut = 0):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQbufferPoolConfig(self._SD_Object__handle, nDAQ, c_void_p(0), nPoints, timeOut, c_void_p(0), c_void_p(0));
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQbufferPoolRelease(self, nDAQ):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_DAQbufferPoolRelease(self._SD_Object__handle, nDAQ);
        else :
            return SD_Error.MODULE_NOT_OPENED;

    def DAQbufferGet(self, nDAQ):
        if self._SD_Object__handle > 0 :
            self._SD_Object__core_dll.SD_AIN_DAQbufferGet.restype = POINTER(c_short)
            error = c_int32()
            readPoints = c_int32()
            data = self._SD_Object__core_dll.SD_AIN_DAQbufferGet(self._SD_Object__handle, nDAQ, byref(readPoints), byref(error))
            error = error.value

            if error < 0 :
                return error
            else :
                nPoints = readPoints.value

                if nPoints > 0 :
                    return np.ctypeslib.as_array((c_short*nPoints).from_address(addressof(data.contents)))
                else :
                    return np.empty(0, dtype=np.short)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def FFT(self, channel, data, dB = False, windowType = 0) :
        error = SD_Error.INVALID_PARAMETERS

        if self._SD_Object__handle > 0 :
            if data is not None :
                size = len(data)

                if size > 0 :
                    resultSize = int(ceil(pow(2, ceil(log(size, 2)))/2))
                    dataC = (c_short * size)(*data)
                    moduleC = (c_double * resultSize)()
                    phaseC = (c_double * resultSize)()

                    resultSize = self._SD_Object__core_dll.SD_AIN_FFT(self._SD_Object__handle, channel, dataC, size, moduleC, resultSize, phaseC, dB, windowType)

                    if resultSize > 0 :
                        moduleData = np.array(moduleC)
                        phaseData = np.array(phaseC)
                    else :
                        moduleData = np.empty(0, dtype=np.double)
                        phaseData = np.empty(0, dtype=np.double)

                    return (moduleData, phaseData)
        else :
            error = SD_Error.MODULE_NOT_OPENED

        return error


    def getAction(self, actionId):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_getAction(self._SD_Object__handle, actionId)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def getEvent(self, eventId):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIN_getEvent(self._SD_Object__handle, eventId)
        else :
            return SD_Error.MODULE_NOT_OPENED

class ActionAio :
    def __init__(self, module):
        self.__module = module
        
    @property
    def ch1_reset_phase(self):
        return self.__module.getAction(constants.SD_AIO_Action_CH1ResetPhase)

    @property
    def ch2_reset_phase(self):
        return self.__module.getAction(constants.SD_AIO_Action_CH2ResetPhase)

    @property
    def ch3_reset_phase(self):
        return self.__module.getAction(constants.SD_AIO_Action_CH3ResetPhase)

    @property
    def ch4_reset_phase(self):
        return self.__module.getAction(constants.SD_AIO_Action_CH4ResetPhase)

    @property
    def awg1_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1Start)

    @property
    def awg2_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2Start)

    @property
    def awg3_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3Start)

    @property
    def awg4_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4Start)

    @property
    def awg1_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1Stop)

    @property
    def awg2_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2Stop)

    @property
    def awg3_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3Stop)

    @property
    def awg4_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4Stop)

    @property
    def awg1_pause(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1Pause)

    @property
    def awg2_pause(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2Pause)

    @property
    def awg3_pause(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3Pause)

    @property
    def awg4_pause(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4Pause)

    @property
    def awg1_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1Resume)

    @property
    def awg2_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2Resume)

    @property
    def awg3_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3Resume)

    @property
    def awg4_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4Resume)

    @property
    def awg1_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1Trigger)

    @property
    def awg2_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2Trigger)

    @property
    def awg3_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3Trigger)

    @property
    def awg4_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4Trigger)

    @property
    def awg1_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1JumpNextWaveform)

    @property
    def awg2_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2JumpNextWaveform)

    @property
    def awg3_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3JumpNextWaveform)

    @property
    def awg4_jump_next_waveform(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4JumpNextWaveform)

    @property
    def awg1_queue_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG1QueueFlush)

    @property
    def awg2_queue_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG2QueueFlush)

    @property
    def awg3_queue_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG3QueueFlush)

    @property
    def awg4_queue_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_AWG4QueueFlush)
    
    #*************DIG************
    
    @property
    def daq1_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ1Start)

    @property
    def daq2_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ2Start)

    @property
    def daq3_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ3Start)

    @property
    def daq4_start(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ4Start)
        
    @property
    def daq1_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ1Stop)

    @property
    def daq2_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ2Stop)

    @property
    def daq3_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ3Stop)

    @property
    def daq4_stop(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ4Stop)
        
    @property
    def daq1_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ1Resume)

    @property
    def daq2_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ2Resume)

    @property
    def daq3_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ3Resume)

    @property
    def daq4_resume(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ4Resume)
    
    @property
    def daq1_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ1Trigger)

    @property
    def daq2_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ2Trigger)

    @property
    def daq3_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ3Trigger)

    @property
    def daq4_trigger(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ4Trigger)
        
    @property
    def daq1_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ1Flush)

    @property
    def daq2_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ2Flush)

    @property
    def daq3_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ3Flush)

    @property
    def daq4_flush(self):
        return self.__module.getAction(constants.SD_AIO_Action_DAQ4Flush)
    
    @property
    def fpga_user_0(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga0)

    @property
    def fpga_user_1(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga1)

    @property
    def fpga_user_2(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga2)

    @property
    def fpga_user_3(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga3)

    @property
    def fpga_user_4(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga4)

    @property
    def fpga_user_5(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga5)

    @property
    def fpga_user_6(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga6)

    @property
    def fpga_user_7(self):
        return self.__module.getAction(constants.SD_AIO_Action_UserFpga7)
        
        
class EventAio:
    def __init__(self, module):
        self.__module = module
        
    @property
    def awg1_queue_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1QueueEmpty)

    @property
    def awg2_queue_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2QueueEmpty)

    @property
    def awg3_queue_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3QueueEmpty)

    @property
    def awg4_queue_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4QueueEmpty)

    @property
    def awg1_queue_full(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1QueueFull)

    @property
    def awg2_queue_full(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2QueueFull)

    @property
    def awg3_queue_full(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3QueueFull)

    @property
    def awg4_queue_full(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4QueueFull)

    @property
    def awg1_underrun(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1Underrun)

    @property
    def awg2_underrun(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2Underrun)

    @property
    def awg3_underrun(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3Underrun)

    @property
    def awg4_underrun(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4Underrun)

    @property
    def awg1_queue_end(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1QueueEnd)

    @property
    def awg2_queue_end(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2QueueEnd)

    @property
    def awg3_queue_end(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3QueueEnd)

    @property
    def awg4_queue_end(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4QueueEnd)

    @property
    def awg1_waveform_start(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1WfStart)

    @property
    def awg2_waveform_start(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2WfStart)

    @property
    def awg3_waveform_start(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3WfStart)

    @property
    def awg4_waveform_start(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4WfStart)

    @property
    def awg1_queue_marker(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1QueueMarker)

    @property
    def awg2_queue_marker(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2QueueMarker)

    @property
    def awg3_queue_marker(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3QueueMarker)

    @property
    def awg4_queue_marker(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4QueueMarker)

    @property
    def awg1_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1QueueFlushed)

    @property
    def awg2_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2QueueFlushed)

    @property
    def awg3_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3QueueFlushed)

    @property
    def awg4_queue_flushed(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4QueueFlushed)

    @property
    def awg1_queue_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1QueueRunning)

    @property
    def awg2_queue_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2QueueRunning)

    @property
    def awg3_queue_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3QueueRunning)

    @property
    def awg4_queue_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4QueueRunning)
    
    #***********DIG*************
    @property
    def daq1_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ1Empty)

    @property
    def daq2_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ2Empty)

    @property
    def daq3_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ3Empty)

    @property
    def daq4_empty(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ4Empty)

    @property
    def daq1_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ1Running)

    @property
    def daq2_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ2Running)

    @property
    def daq3_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ3Running)
    
    @property
    def daq4_running(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ4Running)

    @property
    def daq1_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ1TriggerLoopback)

    @property
    def daq2_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ2TriggerLoopback)

    @property
    def daq3_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ3TriggerLoopback)

    @property
    def daq4_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_DAQ4TriggerLoopback)
    
    @property
    def fpga_user_0(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback0)

    @property
    def fpga_user_1(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback1)

    @property
    def fpga_user_2(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback2)

    @property
    def fpga_user_3(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback3)

    @property
    def fpga_user_4(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback4)

    @property
    def fpga_user_5(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback5)

    @property
    def fpga_user_6(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback6)

    @property
    def fpga_user_7(self):
        return self.__module.getEvent(constants.SD_AIO_Event_UserFpgaLoopback7)

    @property
    def awg1_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG1TriggerLoopback)
    
    @property
    def awg2_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG2TriggerLoopback)
     
    @property
    def awg3_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG3TriggerLoopback)
    
    @property
    def awg4_trigger_loopback(self):
        return self.__module.getEvent(constants.SD_AIO_Event_AWG4TriggerLoopback)
        
        
        
class InstructionAIO(InstructionDIG, InstructionAWG):
    def __init__(self,module):
        self.__module = module
        InstructionDIG.__init__(self,module)
        InstructionAWG.__init__(self,module)


class SD_AIOHvi:
    def __init__(self,module):
        self.__module = module
        self.__engines = Engine(module)
        self.__actions = ActionAio(module)
        self.__events = EventAio(module)
        self.__instructions = InstructionAIO(module)
        self.__triggers = TriggerAIO()
    
    @property
    def engines(self):
        return self.__engines

    @property
    def actions(self):
        return self.__actions

    @property
    def events(self):
        return self.__events
    
    @property
    def instructions(self):
        return self.__instructions
    
    @property
    def triggers(self):
        return self.__triggers

class SD_AIO(SD_AIN,SD_AOU) :
    def __init__(self):
        super(SD_AIO, self).__init__()
        self.__hvi = None

    def createHvi(self):
        self.__hvi = SD_AIOHvi(self)
    
    @property
    def hvi(self):
        return self.__hvi
        
    def getAttributeId64(self, attributeId):
        if self._SD_Object__handle > 0 :
            id = c_longlong(0);           
            self._SD_Object__core_dll.SD_Module_Hvi_getAttributeId64(self._SD_Object__handle, attributeId, byref(id))
            return id.value
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def getAction(self, actionId):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_getAction(self._SD_Object__handle, actionId)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def getEvent(self, eventId):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_getEvent(self._SD_Object__handle, eventId)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def clockIOconfig(self,port,clockConfig):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_clockIOconfig(self._SD_Object__handle, port, clockConfig)
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def clockGetFrequency(self,port):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_clockGetFrequency(self._SD_Object__handle, port)
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def clockGetSyncFrequency(self, port):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_clockGetSyncFrequency(self._SD_Object__handle, port)
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def clockSetFrequency(self, port, frequency, mode = 1):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_clockSetFrequency(self._SD_Object__handle, port, frequency, mode)
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def clockResetPhase(self, port, triggerBehavior, PXItrigger, skew = 0):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_clockResetPhase(self._SD_Object__handle, port, triggerBehavior, PXItrigger, skew)
        else :
            return SD_Error.MODULE_NOT_OPENED

    def triggerIOconfig(self, port, direction):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_triggerIOconfig(self._SD_Object__handle, port, direction)
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def triggerIOwrite(self, port, value, syncMode):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_triggerIOwrite(self._SD_Object__handle, port, value, syncMode)
        else :
            return SD_Error.MODULE_NOT_OPENED
            
    def triggerIOread(self, port):
        if self._SD_Object__handle > 0 :
            return self._SD_Object__core_dll.SD_AIO_triggerIOread(self._SD_Object__handle, port)
        else :
            return SD_Error.MODULE_NOT_OPENED

