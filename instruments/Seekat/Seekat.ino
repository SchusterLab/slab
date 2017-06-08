//Ardunio code for controlling Seekat DC voltage box from openDACS.com.  Designed for Arduino UNO.
//Modified and tested by Gerwin Koolstra, Summer 2016
#include "SPI.h"// necessary library
int sync=10; // using digital pin 10 for SPI slave select
int ldac=9; // Load dac (not implemented). You need to change some jumpers on the boards if you want to use synchronous updating and modify arduino code.
int clr=8; // DAC Clear (not implemented). You need to change some jumpers on the AD58764 boards if you want to use this.
int bytes[9];
int ComLED=1;

void setup()
{
Serial.begin(115200);
Serial.setTimeout(5);
pinMode(5, OUTPUT);//Output communication LED
pinMode(6, OUTPUT);
pinMode(7, OUTPUT); // we use this for SS pin
pinMode(sync, OUTPUT); // we use this for SS pin
pinMode(ldac, OUTPUT); // we use this for SS pin
digitalWrite(7, HIGH);
digitalWrite(sync, HIGH);
SPI.begin(); // wake up the SPI bus.
SPI.setBitOrder(MSBFIRST); //correct order for AD5764.
SPI.setClockDivider(SPI_CLOCK_DIV32);
SPI.setDataMode(SPI_MODE1); //1 and 3 communicate with DAC. 1 is the only one that works with no clock divider.
}
void setValue(int DB[9])
{

 

if (DB[0] == 255&&DB[1]==254&&DB[2]==253) // These bytes serve as a control that communication is working, and are reserved for future functionality such as synchronous updating, clear, and native arduino autoramp.
{
digitalWrite(sync, LOW); //assert sync-bar
int o1 = SPI.transfer(DB[3]); // send command byte to DAC2 in the daisy chain.
Serial.flush();
int o2 = SPI.transfer(DB[4]); // MS data bits, DAC2
Serial.flush();
int o3 = SPI.transfer(DB[5]);//LS 8 data bits, DAC2
Serial.flush();
int o4 = SPI.transfer(DB[6]);// send command byte to DAC1 in the daisy chain.
Serial.flush();
int o5 = SPI.transfer(DB[7]);// MS data bits, DAC1
Serial.flush();
int o6 = SPI.transfer(DB[8]);//LS 8 data bits, DAC1
Serial.flush();
digitalWrite(sync, HIGH);//raise sync-bar to change the dac voltage. Must have LDAC-bar tied low.
Serial.println(o1);
Serial.println(o2);
Serial.println(o3);
Serial.println(o4);
Serial.println(o5);
Serial.println(o6);
Serial.flush();
 switch(ComLED){
  case 0:
  ComLED = 1;
  break;
  case 1:
  ComLED = 0;
  break;
}
}
else //This allows you to check on the scope what has been received by the Arduino for trouble shooting. Use pin 7 to trigger, then look at output of pins 13 (sclk) and 11 on the arduino to readout the bytes the arduino is getting.
{
digitalWrite(7, LOW);
Serial.println(DB[0]);
Serial.println(DB[1]);
Serial.println(DB[2]);
Serial.println(DB[3]);
Serial.println(DB[4]);
Serial.println(DB[5]);
Serial.println(DB[6]);
Serial.println(DB[7]);
Serial.println(DB[8]);
Serial.flush();
//SPI.transfer(DB[0]); Serial.flush();
//SPI.transfer(DB[1]); Serial.flush();
//SPI.transfer(DB[2]); Serial.flush();
//SPI.transfer(DB[3]); Serial.flush();
//SPI.transfer(DB[4]); Serial.flush();
//SPI.transfer(DB[5]); Serial.flush();
//SPI.transfer(DB[6]); Serial.flush();
//SPI.transfer(DB[7]); Serial.flush();
//SPI.transfer(DB[8]); Serial.flush();
digitalWrite(7, HIGH);
}
 
}
void loop()
{
  
switch(ComLED){
  case 0:
  digitalWrite(5, HIGH);//Communication LED "RED"
  digitalWrite(6, LOW);
  break;
  case 1:
  digitalWrite(5, LOW);//Communication LED "GREEN"
  digitalWrite(6, HIGH);
}
if ( Serial.available()) // wait until all data bytes are avaialable
{

for (int i=0; i<9; i++) {
bytes[i] = Serial.parseInt();
//delay(2);
}
if (Serial.available()>0){
int toomuch = Serial.read();
Serial.println(toomuch);}

setValue(bytes);

}
}

