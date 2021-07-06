// include the library code:
#include <Wire.h>

int atten;
char unit_address;
int bitSelect=0;

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.println("Digital Attenator 1");   // send an initial string
    delay(300);
  }
}

void setup_pins() {
  pinMode(8,OUTPUT);
  pinMode(9,OUTPUT);
  pinMode(10,OUTPUT);
  pinMode(11,OUTPUT);
  pinMode(12,OUTPUT);
  pinMode(13,OUTPUT);  
}

void set_attenuator(int value) {
  digitalWrite(13,   !(value & 1));
  digitalWrite(12,  !((value >> 1) & 1));
  digitalWrite(11, !((value >> 2) & 1));
  digitalWrite(10, !((value >> 3) & 1));
  digitalWrite(9, !((value >> 4) & 1));
  digitalWrite(8, !((value >> 5) & 1));  
//  Serial.println(value,DEC);
//  Serial.println(digitalRead(13));
//  Serial.println(digitalRead(12));
//  Serial.println(digitalRead(11));
//  Serial.println(digitalRead(10));
//  Serial.println(digitalRead(9));
//  Serial.println(digitalRead(8));
  
}

void setup() {
  // Debugging output
  Serial.begin(9600);
  // set up the LCD's number of columns and rows: 
 
  // put your setup code here, to run once:
  setup_pins();
  atten=63;
  set_attenuator(atten);
  while (Serial.available() >0) {
  }
  
  Serial.print ('Digital Attenuator ');
  Serial.println (unit_address,DEC);
  //establishContact();  // send a byte to establish contact until receiver responds 
}

uint8_t i=0;
void loop() {
  int cmd;
  int a;

  while (Serial.available() > 0) {
    cmd=toupper(Serial.read());
    //Serial.println(cmd);
    if (cmd == 'S') {
      a = Serial.parseInt();
      //Serial.println(a,DEC);
      if (Serial.read() == '\n') {
        atten=constrain(a,0,63);  
        set_attenuator(atten);
      }
    }
    if (cmd == 'G') {
      Serial.println(atten, DEC);
    }
    if (cmd == 'I') {
      Serial.print("Digital Attenuator ");
      Serial.println(unit_address,DEC); 
    }
    if (cmd == 'A') {
      unit_address = constrain(Serial.parseInt(),0,254);   
    }
  }
 }  
