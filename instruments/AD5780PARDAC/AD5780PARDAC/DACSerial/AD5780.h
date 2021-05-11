/* AD5780.h
 * Created by Larry Chen
 */

#ifndef AD5780_h
#define AD5780_h

#include <Arduino.h>
#include <SPI.h>

/* Read and write bits */
#define READ    0x8
#define WRITE   0x0

/* Register codes */
#define DAC     0x1
#define CTRL    0x2
#define CLR     0x3
#define SCTRL   0x4

#define INIT_CODE0      BITCODE(((WRITE|CTRL)<<4),0x0,0x16) //Clamps output to GND
#define INIT_CODE       BITCODE(((WRITE|CTRL)<<4),0x0,0x12)
/* Given a value to write to the DAC register (0-262143), constructs the 24 bit code to send to the device */
#define DAC_CODE(val)   ((((long)(WRITE|DAC)<<20)&0xF00000)|DAC_TO_BITCODE(val))


/* These macros return the first, second, and third bytes of a 24 bit code */
#define BYTE1(b)   (byte)((b>>16)&0xFF)
#define BYTE2(b)   (byte)((b>>8)&0xFF)
#define BYTE3(b)   (byte)((b&0xFF))


/* Concatenates three bytes into one 24 bit code*/
#define BITCODE(b1, b2, b3) ((((long)b1<<16)|((long)b2<<8)|(long)b3)&0xFFFFFF)
/* Given a 24 bit code, returns the value read from or written to the DAC register (0-262143)*/
#define BITCODE_TO_DAC(b)   ((b>>2)&0x3FFFF)
#define DAC_TO_BITCODE(d)   ((long)(d<<2)&0xFFFFC)


class AD5780
{
    public:
        AD5780(int sync);
        void initialize_DAC();
        void reinitialize_DAC();
        long set_value(long bitcode);
        void set_gcurrval(long input);
        void ramp(long bitcode, long step_size, int delta_t);
        long read_CTRL_register();
        long read_DAC_register();
        long read_CLR_register();
        long read_SCTRL_register();
        long gcurrval = 131072;
    private:
        int _sync;
        long _dac_reg;
        long currval;
        

};

#endif
