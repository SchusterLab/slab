// include the library code:
#include <Wire.h>

int yigval;
char unit_address;
int bitSelect=0;

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.println("YIG controller 1");
    delay(300);
  }
}

void setup_pins() {
  for(int i=2;i<14;i++){
    pinMode(i,OUTPUT);
  }
}

void set_yig(int value) {
  for(int i=2;i<14;i++){
    digitalWrite(i, !((value >> i-2) & 1)); 
  }
}

void setup() {
  // Debugging output
  Serial.begin(9600);
  setup_pins();
  yigval=0;
  set_yig(yigval);
  while (Serial.available() >0) {
  }
  
  Serial.print ('YIG CONTROLLER ');
  Serial.println (unit_address,DEC);
}

uint8_t i=0;
void loop() {
  int cmd;
  int a;
  while (Serial.available() > 0) {
    cmd=toupper(Serial.read());
    if (cmd == 'S') {
      a = Serial.parseInt();
      if (Serial.read() == '\n') {
        yigval=constrain(a,0,4095);  
        set_yig(yigval);
      }
    }
    if (cmd == 'G') {
      Serial.println(yigval, DEC);
    }
    if (cmd == 'I') {
      Serial.print("YIG CONTROLLER ");
      Serial.println(unit_address,DEC); 
    }
    if (cmd == 'A') {
      unit_address = constrain(Serial.parseInt(),0,254);   
    }
  }
 }  
