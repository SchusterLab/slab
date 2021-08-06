#include <SPI.h>
#include <Ethernet2.h>

#include "AD5780.h"

#define HANDLER_ENTRY(NAME, np) { #NAME, handle_ ## NAME, np}

byte mac[] = {
  0x90, 0xA2, 0xDA, 0x10, 0xD0, 0xA8
};

IPAddress ip(192, 168, 14, 158);
IPAddress mydns(192, 168, 14, 1);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);
boolean alreadyConnected = false;

int handle_INIT(int argc, String argv[], EthernetClient client);
int handle_SET(int argc, String argv[], EthernetClient client);
int handle_READ(int argc, String argv[], EthernetClient client);
int handle_RAMP(int argc, String argv[], EthernetClient client);
int handle_ID(int argc, String argv[], EthernetClient client);
int handle_HELP(int argc, String argv[], EthernetClient client);
int handle_ERR(int argc, String argv[], EthernetClient client);
int handle_GETCURRVAL(int argc,String argv[], EthernetClient client);
void process_command(String cmd);

typedef int (*handler_function)(int argc, String argv[], EthernetClient client);

typedef struct handler_entry
{
    String handler_name;
    handler_function func;
    int num_params;
} handler_entry;

handler_entry handlers[] = {
                                HANDLER_ENTRY(INIT, 0), 
                                HANDLER_ENTRY(SET, 2), 
                                HANDLER_ENTRY(READ, 1), 
                                HANDLER_ENTRY(RAMP, 4),
                                HANDLER_ENTRY(ID, 0),
                                HANDLER_ENTRY(HELP, 0),
                                HANDLER_ENTRY(ERR, 0),
                                HANDLER_ENTRY(GETCURRVAL, 0),
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
AD5780 dac8 = AD5780(9);

AD5780 dacs[8] = {dac1, dac2, dac3, dac4, dac5, dac6, dac7,dac8};


int num_dacs = sizeof(dacs) / sizeof(AD5780);

EthernetServer server(23);
void setup() {
  Serial.begin(115200);
  Serial.println("Setting up server");
  Ethernet.begin(mac, ip);
  server.begin();
  Serial.print("Server is at: ");
  Serial.println(Ethernet.localIP());
}

void loop() {
  EthernetClient client = server.available();
  if (client) {
    if (!alreadyConnected) {
      Serial.print("Client connected on socket ");
      Serial.println(client);
      alreadyConnected = true;
    }
    if (client.available() > 0) {
      char received = client.read();
      cmd_str += received;
      if (received == '\n') {
        Serial.print(cmd_str);
        process_command(cmd_str, client);
        cmd_str = "";
      }
    }
  }
}


void process_command(String cmd, EthernetClient client) {
  String argv[10];
  int argc = 0;
  int p1 = 0, p2 = 0;
  int dac = 0;
  long bitcode = 0;
  int increment = 0;
  while((argc < 10) && (cmd[p1] != '\n')) {
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
        handlers[i].func(argc, argv, client);
      } else {
        handle_ERR(argc, argv, client);
      }
      break;
    }
  }
  if (i == num_handlers - 1) {
    handlers[i].func(argc, argv, client);
  }
}


int handle_INIT(int argc, String argv[], EthernetClient client)
{
  //client.println("Initialization");
  if (argc > 1) {
    int dac_num = argv[1].toInt();
    dacs[dac_num - 1].initialize_DAC();
    client.print("Initialized channel ");
    client.println(dac_num);
  } else {
    for (int i = 0; i < num_dacs; i++) {
    dacs[i].initialize_DAC();
    //dacs[i].read_DAC_register();
    }
    client.println("Initialized all channels");
  }
  return 0;
}

int handle_GETCURRVAL(int argc, String argv[], EthernetClient client)
{
  
}
int handle_SET(int argc, String argv[], EthernetClient client)
{
  int dac_num = argv[1].toInt();
  long dac_val = argv[2].toInt();
  long rc = dacs[dac_num - 1].set_value(dac_val);
  dacs[dac_num-1].set_gcurrval(BITCODE_TO_DAC(rc));
  client.println("SET");
  return 0;
}


int handle_READ(int argc, String argv[], EthernetClient client)
{
  //client.println("READ");
  int dac_num = argv[1].toInt();
  long rc = dacs[dac_num - 1].read_DAC_register();
  client.println(BITCODE_TO_DAC(rc));
  return 0;
}


int handle_RAMP(int argc, String argv[], EthernetClient client)
{
  int dac_num = argv[1].toInt();
  long dac_val = argv[2].toInt();//dec value of -10 to 10 range of register
  long step_size = argv[3].toInt();
  int step_time = argv[4].toInt();
  dacs[dac_num - 1].ramp(dac_val, step_size, step_time);
  //instead of reading the register at the end, just convert int to long bitcode string appropriately
  long rc = dacs[dac_num - 1].read_DAC_register();
  dacs[dac_num-1].set_gcurrval(dac_val);//set gcurrval to  value of dac_value
  //client.println(BITCODE_TO_DAC(rc));
  client.println("RAMP");
  return 0;
}


int handle_ID(int argc, String argv[], EthernetClient client)
{
  client.println("AD5780");
  return 0;
}


int handle_HELP(int argc, String argv[], EthernetClient client)
{
  print_help_message(client);
  return 0;
}


int handle_ERR(int argc, String argv[], EthernetClient client)
{
  client.println("Invalid command");
  //print_help_message(client);
  return 0;
}


void print_help_message(EthernetClient client) {
  client.println("Usage: [COMMAND] [PARAM1] [PARAM2] ...");
  client.println("Set voltage: SET [DAC#] [BITCODE]");
  client.println("Ramp voltage: RAMP [DAC#] [BITCODE] [STEP_SIZE] [STEP_TIME]");
  client.println("Read voltage: READ [DAC#]");
  client.println("Initialize dacs: INIT [opt:DAC#]");
  client.println("Help: HELP");
  client.println("1 bit = 38 microVolts");
  client.println("1 milliVolt = 26.2 bits");
}

