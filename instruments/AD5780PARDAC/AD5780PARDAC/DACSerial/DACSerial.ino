#include <SPI.h>
#include "AD5780.h"

#define HANDLER_ENTRY(NAME, np) { #NAME, handle_ ## NAME, np}

int handle_INIT(int argc, String argv[]);
int handle_REINIT(int argc, String argv[]);
int handle_SET(int argc, String argv[]);
int handle_READ(int argc, String argv[]);
int handle_RAMP(int argc, String argv[]);
int handle_PARALLELRAMP(int argc, String argv[]);
int handle_HELP(int argc, String argv[]);
int handle_ERR(int argc, String argv[]);
void process_command(String cmd);

typedef int (*handler_function)(int argc, String argv[]);

typedef struct handler_entry
{
    String handler_name;
    handler_function func;
    int num_params;
} handler_entry;

handler_entry handlers[] = {
                                HANDLER_ENTRY(INIT, 0), 
                                HANDLER_ENTRY(REINIT,0),
                                HANDLER_ENTRY(SET, 2), 
                                HANDLER_ENTRY(READ, 1), 
                                HANDLER_ENTRY(RAMP, 4), 
                                HANDLER_ENTRY(PARALLELRAMP,10),
                                HANDLER_ENTRY(HELP, 0),
                                HANDLER_ENTRY(ERR, 0),
                            };

int num_handlers = sizeof(handlers) / sizeof(handler_entry);

String cmd_str;

AD5780 dac1 = AD5780(2);
AD5780 dac2 = AD5780(3);
AD5780 dac3 = AD5780(4);
AD5780 dac4 = AD5780(5);
AD5780 dac5 = AD5780(6);
AD5780 dac6 = AD5780(7);
AD5780 dac7 = AD5780(8);
AD5780 dac8 = AD5780(13);

AD5780 dacs[8] = {dac1, dac2, dac3, dac4, dac5, dac6, dac7, dac8};

int num_dacs = sizeof(dacs) / sizeof(AD5780);

void setup() {
  SPI.begin();
  pinMode(9, OUTPUT);
  digitalWrite(9, LOW);
  //pinMode(2, OUTPUT);
  //pinMode(3, OUTPUT);
  //pinMode(5, OUTPUT);
  //pinMode(6, OUTPUT);
  Serial.begin(9600);
  Serial.println("Setting up server");
}

void loop() {
  while (Serial.available() > 0) {
    char received = Serial.read();
    cmd_str += received;
    if (received == '\n') {
      Serial.println(cmd_str);
      process_command(cmd_str);
      cmd_str = "";
    }
  }
}


void process_command(String cmd) {
  String argv[11];
  int argc = 0;
  int p1 = 0, p2 = 0;
  int dac = 0;
  long bitcode = 0;
  int increment = 0;
  while((argc < 11) && (cmd[p1] != '\n')) {
    while((cmd[p2] != ' ') && (cmd[p2] != '\n') && (cmd[p2] != '\r')) {
      p2++;
    }
    argv[argc] = cmd.substring(p1,p2);
    argc++;
    if (cmd[p2] == '\n') {
      break;
    }

    p2++;
    p1 = p2;
  }

  int i;
  for (i = 0; i < num_handlers - 1; i++) {
    if (argv[0] == handlers[i].handler_name) {
      if (argc > handlers[i].num_params) {
        handlers[i].func(argc, argv);
      } else {
        handle_ERR(argc, argv);
      }
      break;
    }
  }
  if (i == num_handlers - 1) {
    handlers[i].func(argc, argv);
  }
}

int handle_INIT(int argc, String argv[])
{
  Serial.println("Initialization");
  for (int i = 0; i < num_dacs; i++) {
    dacs[i].initialize_DAC();
    dacs[i].read_DAC_register();
    delay(20);
  }
  Serial.println("Initialization complete");
  return 0;
}

int handle_REINIT(int argc, String argv[])
{
  Serial.println("ReInitialization");
  for (int i=0;i<num_dacs;i++){
    dacs[i].reinitialize_DAC();
    delay(20);
  }
  Serial.println("Reinitialization Complete");
  return 0;
}


int handle_SET(int argc, String argv[])
{
  Serial.println("SET");
  int dac_num = argv[1].toInt();
  long dac_val = argv[2].toInt();
  long rc = dacs[dac_num - 1].set_value(dac_val);
  //Serial.println(BITCODE_TO_DAC(rc));
  Serial.println("SET COMPLETE");
  return 0;
}


int handle_READ(int argc, String argv[])
{
  Serial.println("READ");
  int dac_num = argv[1].toInt();
  long rc = dacs[dac_num - 1].read_DAC_register();
  Serial.println(BITCODE_TO_DAC(rc));
  return 0;
}


int handle_RAMP(int argc, String argv[])
{
  int dac_num = argv[1].toInt();
  long dac_val = argv[2].toInt();
  long step_size = argv[3].toInt();
  int step_time = argv[4].toInt();
  dacs[dac_num - 1].ramp(dac_val, step_size, step_time);
  Serial.println("DAC RAMP DONE");
  return 0;
}

int handle_PARALLELRAMP(int argc,String argv[])
{
  long target_valarray[9];
  long gcurrvalarray[9];
  long n_steparray[9];
  long n_stepmax;
  long step_sizearray[9];
  long step_size = argv[9].toInt();
  long step_time = argv[10].toInt();
  for(int i=0;i<=7;i++){
    target_valarray[i] = argv[i+1].toInt();
//    Serial.println("Target Val: ");
//    Serial.println(String(target_valarray[i]));
    gcurrvalarray[i] = dacs[i].gcurrval;
//    Serial.println("current val: ");
//    Serial.println(String(gcurrvalarray[i]));
    n_steparray[i] = abs(gcurrvalarray[i] - target_valarray[i]) / abs(step_size);
//    Serial.println("n_steps: ");
//    Serial.println(String(n_steparray[i]));
    if (target_valarray[i] > gcurrvalarray[i]) {
      step_sizearray[i] = abs(step_size);
    } else{
      step_sizearray[i] = -abs(step_size);
    }
  }
  //grab max value from n_steparray
  n_stepmax = 0;
  for(int i=0;i<=7;i++){
    if (n_steparray[i] > n_stepmax){
//      Serial.println(n_stepmax);
      n_stepmax = n_steparray[i];
    }
  }
//  Serial.print("Max steps: ");
//  Serial.println(String(n_stepmax));
  for(int i = 0; i < n_stepmax; i++) {
//    Serial.println("Step number: ");
//    Serial.println(String(i));
    for(int j = 0; j <= 7; j++) {
      unsigned long timer = millis();
      while(millis() <= timer + step_time);
      if(i < n_steparray[j]){
        long newval = (gcurrvalarray[j] + i*step_sizearray[j])&0x3FFFF;
        dacs[j].set_value(newval);
      }
    }
  }
  Serial.println("DAC RAMP DONE");
  return 0;
}


int handle_HELP(int argc, String argv[])
{
  print_help_message();
  return 0;
}


int handle_ERR(int argc, String argv[])
{
  Serial.println("Invalid command");
  print_help_message();
  return 0;
}


void print_help_message() {
  Serial.println("Usage: [COMMAND] [PARAM1] [PARAM2] ...");
  Serial.println("Set voltage: SET [DAC#] [BITCODE]");
  Serial.println("Ramp voltage: RAMP [DAC#] [BITCODE] [STEP_SIZE] [STEP_TIME]");
  Serial.println("Read voltage: READ [DAC#]");
  Serial.println("Initialize dacs: INIT [opt:DAC#]");
  Serial.println("ReInitialize dacs: REINIT [opt: DAC#]");
  Serial.println("Help: HELP");
  Serial.println("1 bit = 38 microVolts");
  Serial.println("1 milliVolt = 26.2 bits");
}
