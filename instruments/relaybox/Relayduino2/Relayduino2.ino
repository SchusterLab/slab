#include <SPI.h>

/* KTA_223WebUi.pde -- sample code for KTA-223 using Webduino server library */
// requires Webduino Webserver.h  from http://code.google.com/p/webduino/ to be in \sketchbook\libraries\webduino folder of arduino directory
/*
 * To use this demo,  enter one of the following USLs into your browser.
 * Replace "host" with the IP address assigned to the KTA-223.
 *
 * http://host/demo
 *
 * This URL brings up a display of the status of the Relays, Opto-Ins
 * and analogs.  This is done with a call to defaultCmd.
 * 
 * 
 * http://host/demo/form
 *
 * This URL also brings up a display of the status of the Relays, Opto-Ins
 * and analogs.  But it's done as a form,  by the "formCmd" function,
 * and the Relayss are shown as radio buttons you can change.
 * When you click the "Submit / Refresh" button,  it does a POST that sets the
 * digital pins,  re-reads everything,  and re-displays the form.
 * 
 */

#include "Ethernet.h"
#include "WebServer.h"
#include <stdio.h>
#include <string.h>
#include<stdlib.h>


// no-cost stream operator as described at 
// http://sundial.org/arduino/?page_id=119
template<class T>
inline Print &operator <<(Print &obj, T arg)
{ obj.print(arg); return obj; }


// CHANGE THIS TO YOUR OWN UNIQUE VALUE
static uint8_t mac[] = { 0x90, 0xA2, 0xDA, 0x00, 0x4B, 0x45 };

// CHANGE THIS TO MATCH YOUR HOST NETWORK
static uint8_t ip[] = { 192, 168, 14, 21 };

#define PREFIX "/relaybox"



WebServer webserver(PREFIX, 80);

// commands are functions that get called by the webserver framework
// they can read any posted data from client, and they output to server

void jsonCmd(WebServer &server, WebServer::ConnectionType type,  char* url_tail, bool tail_complete)
{
  
  
  if (type == WebServer::POST)
  {
    server.httpFail();
    return;
  }
  


  server.httpSuccess("application/json");
   
  if(url_tail[0] == 'O' && url_tail[1] == 'N')
   {
     if(url_tail[2] == '0')
     {
       int i;
       for(i=1; i<=8; ++i)
       {
         digitalWrite(i+1, HIGH);
       }
     }
     else if(url_tail[2]=='1' ||url_tail[2]=='2' ||url_tail[2]=='3' ||url_tail[2]=='4'|| url_tail[2]=='5' || url_tail[2]=='6' || url_tail[2]=='7' || url_tail[2]=='8')
     {
       digitalWrite(int(url_tail[2]) - 47, HIGH);
     }
     else
     {
       server << "Please input a relay number.";
     }
   }
  
  else if(url_tail[0] == 'O' && url_tail[1] == 'F')
   {
     if(url_tail[2] == '0')
     {
       int i;
       for(i=1; i<=8; ++i)
       {
         digitalWrite(i+1, LOW);
       }
     }
     else if(url_tail[2]=='1' ||url_tail[2]=='2' ||url_tail[2]=='3' ||url_tail[2]=='4'|| url_tail[2]=='5' || url_tail[2]=='6' || url_tail[2]=='7' || url_tail[2]=='8')
     {
       digitalWrite(int(url_tail[2]) - 47, LOW);
     }
     else
     {
       server << "Please input a relay number.";
     }
   }
  else if(url_tail[0] == 'T' && url_tail[1] == 'R')
    {
      digitalWrite(int(url_tail[2]) - 47, LOW);
      digitalWrite(int(url_tail[2]) - 47, HIGH);
      delay(300);
      digitalWrite(int(url_tail[2]) - 47, LOW);
    }
  else if(url_tail[0] == 'P' && url_tail[1] == 'L')
    {
       if(url_tail[2] == '+')
       {
         if(url_tail[3]=='1' ||url_tail[3]=='2' ||url_tail[3]=='3' ||url_tail[3]=='4'|| url_tail[3]=='5' || url_tail[3]=='6')
         {
           digitalWrite(int(url_tail[3])-47, HIGH);
           digitalWrite(8, HIGH);
           delay(300);
           digitalWrite(8, LOW);
           digitalWrite(int(url_tail[3])-47, LOW);
         }
         else 
         {
           server << "Not a valid input.";
         }
       }
       else if( url_tail[2] =='-')
       {
         if(url_tail[3]=='1' ||url_tail[3]=='2' ||url_tail[3]=='3' ||url_tail[3]=='4'|| url_tail[3]=='5' || url_tail[3]=='6')
         {
           digitalWrite(int(url_tail[3])-47, HIGH);
           digitalWrite(9, HIGH);
           delay(300);
           digitalWrite(9, LOW);
           digitalWrite(int(url_tail[3])-47, LOW);
         }
         else
         {
           server << "Not a valid input.";
         }
       }
    }
 else if(url_tail[0]=='R' && url_tail[1]=='S')
   {
     if(url_tail[2]=='1' ||url_tail[2]=='2' ||url_tail[2]=='3' ||url_tail[2]=='4'|| url_tail[2]=='5' || url_tail[2]=='6' ||url_tail[2]=='7'||url_tail[2]=='8')
       {
         int val = digitalRead(int(url_tail[3])-47));
         server << "Status of Relay " << int(url_tail[3])-48)<< ":" << val;
         return;
       }
     else if(url_tail[3]=='0')
       {
         int i;    
        server << "Status of Relays" << ":";
        for (i = 1; i <= 8; ++i)
        {
            // ignore the pins we use to talk to the Ethernet chip
             int val = digitalRead(i+1);
              server << "Pin" << i << ":" << val << ", ";
        }
        return;
      }
      
    else
      {
         server<< "Not a valid input";
      }
  }
 else if(url_tail[0]=='A' && url_tail[1]=='I')
  {
    if(url_tail[2]=='1' ||url_tail[2]=='2' ||url_tail[2]=='3' ||url_tail[2]=='4'|| url_tail[2]=='5' || url_tail[2]=='6' ||url_tail[2]=='7'||url_tail[2]=='8')
      {
         int val = analogRead(int(url_tail[3])-49));
         server << "Status of Relay " << int(url_tail[3])-48)<< ":" << val;
         return;
       }
  else if(url_tail[3]=='0')
       {
         int i;    
        server << "Status of Relays" << ":";
        for (i = 0; i <= 8; ++i)
        {
            // ignore the pins we use to talk to the Ethernet chip
             int val = analogRead(i);
              server << "Pin" << i << ":" << val << ", ";
        }
        return;
      }
      
    else
      {
         server<< "Not a valid input";
      }
  }
 
 else
    {
      server << "Not a valid command. Currently only valid commands are ON, OF, and TR followed by the relay number.";
    }

  int i;    
  server << "Status of Relays" << "{ ";
  for (i = 1; i <= 8; ++i)
  {
    // ignore the pins we use to talk to the Ethernet chip
    int val = digitalRead(i+1);
    server << "Pin" << i << ":" << val << ", ";
  }
  
  
  server << " }";

 
   
   


 
  server << "done!";
  
 

 
}

void outputPins(WebServer &server, WebServer::ConnectionType type, bool addControls = false)
{
  int val;
  P(htmlHead) =
    "<html>"
    "<head>"
    "<title>Arduino Web Server</title>"
    "<style type=\"text/css\">"
    "BODY { font-family: sans-serif }"
    "H1 { font-size: 14pt; text-decoration: underline }"
    "H2 { font-size: 16pt; text-decoration: underline }"
    "P  { font-size: 10pt; }"
    "</style>"
    "</head>"
    "<body>";

  int i;
  server.httpSuccess();
  server.printP(htmlHead);

  if (addControls)
    server << "<form action='" PREFIX "/form' method='post'>";
  server << "<h2>Relayduino</h1><p>";
  server << "<h1>Relays</h1><p>";

  for (i = 1; i <= 6; ++i)
  {
    // ignore the pins we use to talk to the Ethernet chip
    val = digitalRead(i+1);
    server << "Relay " << i << ": ";
    if (addControls)
    {
      char pinName[4];
      pinName[0] = 'R';
      itoa(i, pinName + 1, 10);
      server.radioButton(pinName, "1", "On", val);
      server << " ";
      server.radioButton(pinName, "0", "Off", !val);
    }
    else
      server << (val ? "ON" : "OFF");

    server << "<br/>";
  }
  
  for (i = 7; i <= 8; ++i)
  {
    // ignore the pins we use to talk to the Ethernet chip
    val = digitalRead(i+1);
    server << "Relay " << i << ": ";
    if (addControls)
    {
      char pinName[4];
      pinName[0] = 'D';
      itoa(i, pinName + 1, 10);
      server.radioButton(pinName, "1", "Pulse", val);
      server << " ";
      server.radioButton(pinName, "0", "Off", !val);
    }
    else
      server << (val ? "ON" : "OFF");

    server << "<br/>";
  }

  server << "</p><h1>Opto-Ins</h1><p>";
  for (i = 1; i <= 4; ++i)
  {
    val = digitalRead(i+14);
    server << "Opto-In " << i << ": ";
    if (val==1) server << "OFF <br/>";
    else server << "ON <br/>";
  }
  
  server << "</p><h1>Analogs</h1><p>";
  for (i = 1; i <= 3; ++i)
  {
    
    if (i==1) val = analogRead(6);
    if (i==2) val = analogRead(7);
    if (i==3) val = analogRead(0);
    server << "Analog " << i << ": " << val << "<br/>";
  }
  server << "</p>";
  

  if (addControls)
    server << "<input type='submit' value='Submit / Refresh'/></form>";

  server << "</body></html>";
}

void formCmd(WebServer &server, WebServer::ConnectionType type, char *url_tail, bool tail_complete)
{
  if (type == WebServer::POST)
  {
    bool repeat;
    char name[16], value[16];
    do
    {
      repeat = server.readPOSTparam(name, 16, value, 16);
      if (name[0] == 'R')
      {
        int pin = strtoul(name + 1, NULL, 10);
        int val = strtoul(value, NULL, 10);
        digitalWrite(pin+1, val);
      }
      if (name[0] == 'D')
      {
        int pin = strtoul(name + 1, NULL, 10);
        int val = strtoul(value, NULL, 10);
        
      }
    } while (repeat);

    server.httpSeeOther(PREFIX "/form");
  }
  else
    outputPins(server, type, true);
}

void defaultCmd(WebServer &server, WebServer::ConnectionType type, char *url_tail, bool tail_complete)
{
  outputPins(server, type, false);  
}

void setup()
{
  // set pins 2-9 for digital input
  for (int i = 2; i <= 9; ++i)
    pinMode(i, OUTPUT);
  

  Ethernet.begin(mac, ip);
  webserver.begin();

  webserver.setDefaultCommand(&defaultCmd);
  webserver.addCommand("json", &jsonCmd);
  webserver.addCommand("form", &formCmd);
}

void loop()
{
  // process incoming connections one at a time forever
  webserver.processConnection();

  // if you wanted to do other work based on a connecton, it would go here
}
