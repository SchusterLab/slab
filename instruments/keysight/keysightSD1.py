import os;
from ctypes import *
from math import pow, log, ceil

import numpy as np

class SD_Object :
	__core_dll = cdll.LoadLibrary("SD1core" if os.name == 'nt' else "libSD1core.so")

	def __init__(self) :
		self.__handle = 0;

	@classmethod
	def __formatString(cls, string) :
		tmp = string.decode();
		return tmp[0:tmp.find('\0')];

class SD_Error(SD_Object) :
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
	END = 4;

class SD_TriggerValue :
	LOW = 0;
	HIGH = 1;

class SD_SyncModes :
	SYNC_NONE = 0;
	SYNC_CLK10 = 1;

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

class SD_TriggerModes :
	AUTOTRIG = 0;
	VIHVITRIG = 1;
	SWHVITRIG = 1;
	EXTTRIG = 2;
	ANALOGTRIG = 3;
	SWHVITRIG_CYCLE = 5;
	EXTTRIG_CYCLE = 6;
	ANALOGAUTOTRIG = 7;

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

class SD_DIO_Bus :
	DIO_INPUT_BUS0 = 1000;
	DIO_INPUT_BUS1 = 1001;
	DIO_OUTPUT_BUS0 = 2000;
	DIO_OUTPUT_BUS1 = 2001;

class SD_Compatibility :
	LEGACY = 0;
	KEYSIGHT = 1;

class SD_Wave(SD_Object) :
	PADDING_ZERO = 0;
	PADDING_REPEAT = 1;

	def newFromFile(self, waveformFile) :
		self._SD_Object__handle = self._SD_Object__core_dll.SD_Wave_newFromFile(waveformFile.encode());

		return self._SD_Object__handle;

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
			return 1;
		else :
			return SD_Error.CREATING_WAVE;

	def getType(self) :
		return SD_Object_Type.WAVE;

class SD_Module(SD_Object) :
	def openWithSerialNumber(self, partNumber, serialNumber) :
		if self._SD_Object__handle <= 0 :
			self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSerialNumber(partNumber.encode(), serialNumber.encode())

		return self._SD_Object__handle;

	def openWithSerialNumberCompatibility(self, partNumber, serialNumber, compatibility) :
		if self._SD_Object__handle <= 0 :
			self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSerialNumberCompatibility(partNumber.encode(), serialNumber.encode(), compatibility);

		return self._SD_Object__handle;

	def openWithSlot(self, partNumber, nChassis, nSlot) :
		if self._SD_Object__handle <= 0 :
			self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSlot(partNumber.encode(), nChassis, nSlot)

		return self._SD_Object__handle;

	def openWithSlotCompatibility(self, partNumber, nChassis, nSlot, compatibility):
		if self._SD_Object__handle <= 0:
			self._SD_Object__handle = self._SD_Object__core_dll.SD_Module_openWithSlotCompatibility(partNumber.encode(), nChassis, nSlot, compatibility);

		return self._SD_Object__handle;

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
		buffer = ''.rjust(20, '\0').encode();

		error = cls._SD_Object__core_dll.SD_Module_getSerialNumberBySlot(chassis, slot, buffer);

		if error < 0 :
			return error;
		else :
			return cls._SD_Object__formatString(buffer);

	@classmethod
	def getSerialNumberByIndex(cls, index) :
		buffer = ''.rjust(20, '\0').encode();

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

	def getStatus(self) :
		result = 0;

		if self._SD_Object__handle > 0 :
			result = self._SD_Object__core_dll.SD_Module_getStatus(self._SD_Object__handle);
		else :
			result = SD_Error.MODULE_NOT_OPENED;

		return result;

	def runSelfTest(self) :
		result = 0;

		if self._SD_Object__handle > 0 :
			result = self._SD_Object__core_dll.SD_Module_runSelfTest(self._SD_Object__handle);
		else :
			result = SD_Error.MODULE_NOT_OPENED;

		return result;

	def getSerialNumber(self) :
		serial = ''.rjust(20, '\0').encode();

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
		if self._SD_Object__handle > 0 :
			self._SD_Object__core_dll.SD_Module_getFirmwareVersion.restype = c_double;
			result = self._SD_Object__core_dll.SD_Module_getFirmwareVersion(self._SD_Object__handle);

			if result < 0 :
				return int(result);
			else :
				return result;
		else :
			return  SD_Error.MODULE_NOT_OPENED;

	def getHardwareVersion(self) :
		if self._SD_Object__handle > 0 :
			self._SD_Object__core_dll.SD_Module_getHardwareVersion.restype = c_double;
			result = self._SD_Object__core_dll.SD_Module_getHardwareVersion(self._SD_Object__handle);

			if result < 0 :
				return int(result);
			else :
				return result;
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

	##HVI Registers
	def readRegisterByNumber(self, varNumber) :
		varValue = 0;

		if self._SD_Object__handle > 0 :
			error = c_int32();
			varValue = self._SD_Object__core_dll.SD_Module_readRegister(self._SD_Object__handle, varNumber, byref(error));
			error = error.value;
		else :
			error = SD_Error.MODULE_NOT_OPENED;

		return (error, varValue);

	def readRegisterByName(self, varName) :
		varValue = 0;

		if self._SD_Object__handle > 0 :
			error = c_int32();
			varValue = self._SD_Object__core_dll.SD_Module_readRegisterWithName(self._SD_Object__handle, varName.encode(), byref(error));
			error = error.value;
		else :
			error = SD_Error.MODULE_NOT_OPENED;

		return (error, varValue);

	def readRegisterDoubleByNumber(self, varNumber, unit) :
		varValue = 0;

		if self._SD_Object__handle > 0 :
			error = c_int32();
			self._SD_Object__core_dll.SD_Module_readDoubleRegister.restype = c_double;
			varValue = self._SD_Object__core_dll.SD_Module_readDoubleRegister(self._SD_Object__handle, varNumber, unit.encode(), byref(error));
			error = error.value;
		else :
			error = SD_Error.MODULE_NOT_OPENED;

		return (error, varValue);

	def readRegisterDoubleByName(self, varName, unit) :
		varValue = 0;

		if self._SD_Object__handle > 0 :
			error = c_int32();
			self._SD_Object__core_dll.SD_Module_readDoubleRegisterWithName.restype = c_double;
			varValue = self._SD_Object__core_dll.SD_Module_readDoubleRegisterWithName(self._SD_Object__handle, varName.encode(), unit.encode(), byref(error));
			error = error.value;
		else :
			error = SD_Error.MODULE_NOT_OPENED;

		return (error, varValue);

	def writeRegisterByNumber(self, varNumber, varValue) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_writeRegister(self._SD_Object__handle, varNumber, varValue);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def writeRegisterByName(self, varName, varValue) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_writeRegisterWithName(self._SD_Object__handle, varName.encode(), varValue);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def writeRegisterDoubleByNumber(self, varNumber, value, unit) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_writeDoubleRegister(self._SD_Object__handle, varNumber, c_double(value), unit.encode());
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def writeRegisterDoubleByName(self, varName, value, unit) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_writeDoubleRegisterWithName(self._SD_Object__handle, varName.encode(), c_double(value), unit.encode());
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
	def FPGAreadPCport(self, port, nDW, address, addressMode, accessMode) :
		if self._SD_Object__handle > 0 :
			if nDW > 0 :
				data = (c_int * int(nDW))()
				error = self._SD_Object__core_dll.SD_Module_FPGAreadPCport(self._SD_Object__handle, port, data, nDW, address, addressMode, accessMode)

				if error < 0 :
					return error
				else :
					return np.array(data)
			else :
				return SD_Error.INVALID_VALUE
		else :
			return SD_Error.MODULE_NOT_OPENED

	def FPGAwritePCport(self, port, buffer, address, addressMode, accessMode) :
		if self._SD_Object__handle > 0 :
			if len(buffer) > 0 :
				data = (c_int * len(buffer))(*buffer);
				return self._SD_Object__core_dll.SD_Module_FPGAwritePCport(self._SD_Object__handle, port, data, data._length_, address, addressMode, accessMode);
			else :
				return SD_Error.INVALID_VALUE;
		else :
			return SD_Error.MODULE_NOT_OPENED;

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

	def openHVI(self, fileHVI) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_openHVI(self._SD_Object__handle, fileHVI.encode());
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def compileHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_compileHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def compilationErrorMessageHVI(self, errorIndex) :
		error = 0;
		message = ''.rjust(200, '\0').encode();

		if self._SD_Object__handle > 0 :
			error = self._SD_Object__core_dll.SD_Module_compilationErrorMessageHVI(self._SD_Object__handle, errorIndex, message, len(message));

			if error >= 0 :
				return self._SD_Object__formatString(message);
		else :
			error = SD_Error.MODULE_NOT_OPENED;

		return error;

	def loadHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_loadHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	##HVI Control
	def startHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_startHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def pauseHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_pauseHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def resumeHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_resumeHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def stopHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_stopHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def resetHVI(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_Module_resetHVI(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

class SD_AOU(SD_Module):
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

	def triggerIOconfigV5(self, direction, syncMode = 1) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_AOU_triggerIOconfigV5(self._SD_Object__handle, direction, syncMode);
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
				dataC = (c_double * len(dataRaw))(*dataRaw);
				return self._SD_Object__core_dll.SD_AOU_waveformLoadArrayInt16(self._SD_Object__handle, waveformType, dataC._length_, dataC, waveformNumber, paddingMode);
			else :
				return SD_Error.INVALID_VALUE;
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

	def AWGfromArray(self, nAWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveformDataA, waveformDataB = None, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			if len(waveformDataA) > 0 and (waveformDataB is None or len(waveformDataA) == len(waveformDataB)) :
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

	def AWGidleValue(self, nAWG, value) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_AOU_AWGidleValue(self._SD_Object__handle, nAWG, c_double(value));
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def AWGidleValueRead(self, nAWG) :
		if self._SD_Object__handle > 0 :
			self._SD_Object__core_dll.SD_AOU_AWGidleValueRead.restype = c_double;
			return self._SD_Object__core_dll.SD_AOU_AWGidleValueRead(self._SD_Object__handle, nAWG);
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

class SD_DIO(SD_Module) :
	##Config
	def IOstandardConfig(self, portSector, logicStandard) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_IOstandardConfig(self._SD_Object__handle, portSector, logicStandard);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def IOdirectionConfig(self, lineMask, direction) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_IOdirectionConfig(self._SD_Object__handle, c_longlong(lineMask), direction);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	##Ports
	def portWrite(self, nPort, portValue) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_portWrite(self._SD_Object__handle, nPort, c_longlong(portValue));
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def portWriteWithMask(self, nPort, portValue, lineMask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_portWriteWithMask(self._SD_Object__handle, nPort, c_longlong(portValue), c_longlong(lineMask));
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def portRead(self, nPort) :
		value = c_longlong(0);
		error = c_int(SD_Error.MODULE_NOT_OPENED);

		if self._SD_Object__handle > 0 :
			self._SD_Object__core_dll.SD_DIO_portRead.restype = c_longlong;
			value = self._SD_Object__core_dll.SD_DIO_portRead(self._SD_Object__handle, nPort, byref(error));

		return (value.value, error.value);

	##Buses
	def busConfig(self, nBus, nPort, StartBit, EndBit):
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_busConfig(self._SD_Object__handle, nBus, nPort, StartBit, EndBit);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def busWrite(self, nBus, busValue) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_busWrite(self._SD_Object__handle, nBus, busValue);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def busRead(self, nBus) :
		error = c_int(SD_Error.MODULE_NOT_OPENED);
		value = 0;

		if self._SD_Object__handle > 0 :
			value = self._SD_Object__core_dll.SD_DIO_busRead(self._SD_Object__handle, nBus, byref(error));

		return (value, error.value);

	def busSamplingConfig(self, nBus, switchStrobe, strobeOn, strobeType, strobeDelay, prescaler = 0, debouncing = 0) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_busSamplingConfig(self._SD_Object__handle, nBus, switchStrobe, strobeOn, strobeType, c_float(strobeDelay), prescaler, debouncing);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	##Digital Lines
	def lineWrite(self, nPort, nLine, lineValue) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_lineWrite(self._SD_Object__handle, nPort, nLine, lineValue);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def lineRead(self, nPort, nLine) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_lineRead(self._SD_Object__handle, nPort, nLine);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	##DWG
	def waveformGetAddress(self, waveformNumber) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformGetAddress(self._SD_Object__handle, waveformNumber);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformGetMemorySize(self, waveformNumber) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformGetMemorySize(self._SD_Object__handle, waveformNumber);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformMemoryGetWriteAddress(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformMemoryGetWriteAddress(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformMemorySetWriteAddress(self, writeAddress) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformMemorySetWriteAddress(self._SD_Object__handle, writeAddress);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformReLoad(self, waveformObject, waveformNumber, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformReLoad(self._SD_Object__handle, waveformObject._SD_Object__handle, waveformNumber, paddingMode);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformReLoadArrayInt16(self, waveformType, dataRaw, waveformNumber, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			if len(dataRaw) > 0 :
				dataC = (c_short * len(dataRaw))(*dataRaw);
				return self._SD_Object__core_dll.SD_DIO_waveformReLoadArrayInt16(self._SD_Object__handle, waveformType, dataC._length_, dataC, waveformNumber, paddingMode);
			else :
				return SD_Error.INVALID_VALUE;
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformLoad(self, waveformObject, waveformNumber, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformLoad(self._SD_Object__handle, waveformObject._SD_Object__handle, waveformNumber, paddingMode);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformLoadArrayInt16(self, waveformType, dataRaw, waveformNumber, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			if len(dataRaw) > 0 :
				dataC = (c_short * len(dataRaw))(*dataRaw);
				return self._SD_Object__core_dll.SD_DIO_waveformLoadArrayInt16(self._SD_Object__handle, waveformType, dataC._length_, dataC, waveformNumber, paddingMode);
			else :
				return SD_Error.INVALID_VALUE;
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def waveformFlush(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_waveformFlush(self._SD_Object__handle);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGqueueWaveform(self, nDWG, waveformNumber, triggerMode, startDelay, cycles, prescaler) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGqueueWaveform(self._SD_Object__handle, nDWG, waveformNumber, triggerMode, startDelay, cycles, prescaler);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGstart(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGstart(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGstop(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGstop(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGresume(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGresume(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGpause(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGpause(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGtrigger(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGtrigger(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGstartMultiple(self, DWGmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGstartMultiple(self._SD_Object__handle, DWGmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGstopMultiple(self, DWGmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGstopMultiple(self._SD_Object__handle, DWGmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGresumeMultiple(self, DWGmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGresumeMultiple(self._SD_Object__handle, DWGmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGpauseMultiple(self, DWGmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGpauseMultiple(self._SD_Object__handle, DWGmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGtriggerMultiple(self, DWGmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGtriggerMultiple(self._SD_Object__handle, DWGmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGflush(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGflush(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGisRunning(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGisRunning(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGnWFplaying(self, nDWG) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGnWFplaying(self._SD_Object__handle, nDWG);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWGfromFile(self, nDWG, waveformFile, triggerMode, startDelay, cycles, prescaler, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGfromFile(self._SD_Object__handle, nDWG, waveformFile.encode(), triggerMode, startDelay, cycles, prescaler, paddingMode);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DWG(self, nDWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveformDataA, waveformDataB = None, paddingMode = 0) :
		if self._SD_Object__handle > 0 :
			if len(waveformDataA) > 0 and (waveformDataB is None or len(waveformDataA) == len(waveformDataB)) :
				waveform_dataA_C = (c_int * len(waveformDataA))(*waveformDataA);

				if waveformDataB is None:
					waveform_dataB_C = c_void_p(0);
				else :
					waveform_dataB_C = (c_int * len(waveformDataB))(*waveformDataB);

				return self._SD_Object__core_dll.SD_DIO_DWGfromArray(self._SD_Object__handle, nDWG, triggerMode, startDelay, cycles, prescaler, waveformType, waveform_dataA_C._length_, waveform_dataA_C, waveform_dataB_C, paddingMode)
			else :
				return SD_Error.INVALID_VALUE
		else :
			return SD_Error.MODULE_NOT_OPENED

	def DWGtriggerExternalConfig(self, nDWG, externalSource, triggerBehavior) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DWGtriggerExternalConfig(self._SD_Object__handle, nDWG, externalSource, triggerBehavior);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQread(self, nDAQ, nPoints, timeOut = 0) :
		if self._SD_Object__handle > 0 :
			if nPoints > 0 :
				data = (c_short * nPoints)()

				nPoints = self._SD_Object__core_dll.SD_DIO_DAQread(self._SD_Object__handle, nDAQ, data, nPoints, timeOut)

				if nPoints > 0 :
					return np.array(data)
				else :
					return np.empty(0, dtype=np.short)
			else :
				return SD_Error.INVALID_VALUE
		else :
			return SD_Error.MODULE_NOT_OPENED

	def DAQconfig(self, nDAQ, nDAQpointsPerCycle, nCycles, prescaler, triggerMode) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQconfig(self._SD_Object__handle, nDAQ, nDAQpointsPerCycle, nCycles, prescaler, triggerMode);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQtriggerExternalConfig(self, nDAQ, externalSource, triggerBehavior) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQtriggerExternalConfig(self._SD_Object__handle, nDAQ, externalSource, triggerBehavior);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQcounterRead(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQcounterRead(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQstart(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQstart(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQpause(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQpause(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQresume(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQresume(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQflush(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQflush(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQstop(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQstop(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQtrigger(self, nDAQ) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQtrigger(self._SD_Object__handle, nDAQ);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQstartMultiple(self, DAQmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQstartMultiple(self._SD_Object__handle, DAQmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQpauseMultiple(self, DAQmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQpauseMultiple(self._SD_Object__handle, DAQmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQresumeMultiple(self, DAQmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQresumeMultiple(self._SD_Object__handle, DAQmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQflushMultiple(self, DAQmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQflushMultiple(self._SD_Object__handle, DAQmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQstopMultiple(self, DAQmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQstopMultiple(self._SD_Object__handle, DAQmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

	def DAQtriggerMultiple(self, DAQmask) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_DIO_DAQtriggerMultiple(self._SD_Object__handle, DAQmask);
		else :
			return SD_Error.MODULE_NOT_OPENED;

class SD_AIN_TriggerMode :
	RISING_EDGE = 1;
	FALLING_EDGE = 2;
	BOTH_EDGES = 3;

class SD_AIN_SyncTriggerBehaviours :
	TRIGGER_HIGH_SYNC = SD_TriggerBehaviors.TRIGGER_HIGH + 8;
	TRIGGER_LOW_SYNC = SD_TriggerBehaviors.TRIGGER_LOW + 8;
	TRIGGER_RISE_SYNC = SD_TriggerBehaviors.TRIGGER_RISE + 8;
	TRIGGER_FALL_SYNC = SD_TriggerBehaviors.TRIGGER_FALL + 8;

class AIN_Coupling :
	AIN_COUPLING_DC = 0;
	AIN_COUPLING_AC = 1;

class AIN_Impedance :
	AIN_IMPEDANCE_HZ = 0;
	AIN_IMPEDANCE_50 = 1;

class SD_AIN(SD_Module) :
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

	def DAQstart(self, channel) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_AIN_DAQstart(self._SD_Object__handle, channel);
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

				nPoints = self._SD_Object__core_dll.SD_AIN_DAQread(self._SD_Object__handle, nDAQ, data, nPoints, timeOut)

				if nPoints > 0 :
					return np.array(data)
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
					return  np.array(data)
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

class SD_HVI(SD_Object) :
	def isOpen(self) :
		return (self._SD_Object__handle > 0);

	def open(self, fileHVI) :
		status = self._SD_Object__handle;

		if status <= 0 :
			status = self._SD_Object__handle = self._SD_Object__core_dll.SD_HVI_open(fileHVI.encode());

			if status > 0 :
				status = self._SD_Object__core_dll.SD_HVI_load(self._SD_Object__handle);

		return status;

	def close(self) :
		if self._SD_Object__handle > 0 :
			self._SD_Object__handle = self._SD_Object__core_dll.SD_HVI_close(self._SD_Object__handle);

		return self._SD_Object__core_dll;

	def getType(self) :
		return SD_Object_Type.HVI;

	##HVI Control
	def compile(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_compile(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def compilationErrorMessage(self, errorIndex) :
		error = 0;
		message = ''.rjust(200, '\0').encode();

		if self._SD_Object__handle > 0 :
			error = self._SD_Object__core_dll.SD_HVI_compilationErrorMessage(self._SD_Object__handle, errorIndex, message, len(message));

			if error >= 0 :
				return self._SD_Object__formatString(message);
		else :
			error = SD_Error.HVI_NOT_OPENED;

		return error;

	def load(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_load(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def assignHardwareWithIndexAndSerialNumber(self, index, partNumber, serialNumber) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_assignHardwareWithIndexAndSerialNumber(self._SD_Object__handle, index, partNumber.encode(), serialNumber.encode());
		else :
			return SD_Error.HVI_NOT_OPENED;

	def assignHardwareWithIndexAndSlot(self, index, nChassis, nSlot) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_assignHardwareWithIndexAndSlot(self._SD_Object__handle, index, nChassis, nSlot);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def assignHardwareWithUserNameAndSerialNumber(self, moduleUserName, partNumber, serialNumber) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_assignHardwareWithUserNameAndSerialNumber(self._SD_Object__handle, moduleUserName.encode(), partNumber.encode(), serialNumber.encode());
		else :
			return SD_Error.HVI_NOT_OPENED;

	def assignHardwareWithUserNameAndSlot(self, moduleUserName, nChassis, nSlot) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_assignHardwareWithUserNameAndSlot(self._SD_Object__handle, moduleUserName.encode(), nChassis, nSlot);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def assignHardwareWithUserNameAndModuleID(self, moduleUserName, module) :
		if self._SD_Object__handle > 0 :
			if module is not None and module.isOpen() :
				return self._SD_Object__core_dll.SD_HVI_assignHardwareWithUserNameAndModuleID(self._SD_Object__handle, moduleUserName.encode(), module._SD_Object__handle);
			else :
				return SD_Error.MODULE_NOT_OPENED;
		else :
			return SD_Error.HVI_NOT_OPENED;

	def start(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_start(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def pause(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_pause(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def resume(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_resume(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def stop(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_stop(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def reset(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_reset(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	##HVI Modules
	def getNumberOfModules(self) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_getNumberOfModules(self._SD_Object__handle);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def getModuleName(self, index) :
		if self._SD_Object__handle > 0 :
			self._SD_Object__core_dll.SD_HVI_getModuleNameHidden.restype = c_char_p;
			return self._SD_Object__core_dll.SD_HVI_getModuleNameHidden(self._SD_Object__handle, index).decode();
		else :
			return SD_Error.MODULE_NOT_FOUND;

	def getModuleIndex(self, moduleUserName) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_getModuleIndex(self._SD_Object__handle, moduleUserName.encode());
		else :
			return SD_Error.HVI_NOT_OPENED;

	def getModuleByIndex(self, index) :
		if self._SD_Object__handle > 0 :
			return self.getModuleByName(self.getModuleName(index));
		else :
			return SD_Error.HVI_NOT_OPENED;

	def getModuleByName(self, moduleUserName) :
		if self._SD_Object__handle > 0 :
			moduleHandle = self._SD_Object__core_dll.SD_HVI_getModuleIDwithUserName(self._SD_Object__handle, moduleUserName.encode());

			if moduleHandle > 0 :
				switcher = {
					SD_Object_Type.AOU: SD_AOU(),
					SD_Object_Type.DIO: SD_DIO(),
##					SD_Object_Type.TDC: SD_TDC(),
					SD_Object_Type.AIN: SD_AIN(),
##					SD_Object_Type.AIO: SD_AIO(),
				}

				requestModule = switcher.get(SD_Module.getType(moduleHandle), "nothing");

				if requestModule == "nothing" :
					return SD_Error.MODULE_NOT_FOUND;
				else :
					requestModule._SD_Object__handle = moduleHandle;
			else :
				return moduleHandle;
		else :
			return SD_Error.HVI_NOT_OPENED;

	##HVI Module's Constants
	def readIntegerConstantWithIndex(self, moduleIndex, constantName) :
		value = c_int32();
		error = SD_Error.HVI_NOT_OPENED;

		if self._SD_Object__handle > 0 :
			error = self._SD_Object__core_dll.SD_HVI_readIntegerConstantWithIndex(self._SD_Object__handle, moduleIndex, constantName.encode(), byref(value));

		return (error, value.value);

	def readIntegerConstantWithUserName(self, moduleUserName, constantName) :
		value = c_int32();
		error = SD_Error.HVI_NOT_OPENED;

		if self._SD_Object__handle > 0 :
			error = self._SD_Object__core_dll.SD_HVI_readIntegerConstantWithUserName(self._SD_Object__handle, moduleUserName.encode(), constantName.encode(), byref(value));

		return (error, value.value);

	def writeIntegerConstantWithIndex(self, moduleIndex, constantName, value) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_writeIntegerConstantWithIndex(self._SD_Object__handle, moduleIndex, constantName.encode(), value);
		else: 
			return SD_Error.HVI_NOT_OPENED;

	def writeIntegerConstantWithUserName(self, moduleUserName, constantName, value) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_writeIntegerConstantWithUserName(self._SD_Object__handle, moduleUserName.encode(), constantName.encode(), value);
		else :
			return SD_Error.HVI_NOT_OPENED;

	def readDoubleConstantWithIndex(self, moduleIndex, constantName) :
		value = c_double();
		unit = c_char_p();
		error = SD_Error.HVI_NOT_OPENED;

		if self._SD_Object__handle > 0 :
			error = self._SD_Object__core_dll.SD_HVI_readDoubleConstantWithIndex(self._SD_Object__handle, moduleIndex, constantName.encode(), byref(value), byref(unit));

		return (error, value.value, unit.value.decode());

	def readDoubleConstantWithUserName(self, moduleUserName, constantName) :
		value = c_double();
		unit = c_char_p();
		error = SD_Error.HVI_NOT_OPENED;

		if self._SD_Object__handle > 0 :
			error = self._SD_Object__core_dll.SD_HVI_readDoubleConstantWithUserName(self._SD_Object__handle, moduleUserName.encode(), constantName.encode(), byref(value), byref(unit));

		return (error, value.value, unit.value.decode());

	def writeDoubleConstantWithIndex(self, moduleIndex, constantName, value, unit) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_writeDoubleConstantWithIndex(self._SD_Object__handle, moduleIndex, constantName.encode(), c_double(value), unit.encode());
		else :
			return SD_Error.HVI_NOT_OPENED;

	def writeDoubleConstantWithUserName(self, moduleUserName, constantName, value, unit) :
		if self._SD_Object__handle > 0 :
			return self._SD_Object__core_dll.SD_HVI_writeDoubleConstantWithUserName(self._SD_Object__handle, moduleUserName.encode(), constantName.encode(), c_double(value), unit.encode());
		else :
			return SD_Error.HVI_NOT_OPENED;
