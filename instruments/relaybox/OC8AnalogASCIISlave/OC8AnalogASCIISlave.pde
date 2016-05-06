

/*
 * OC RelayDuino 8 Analog ASCII Slave
 * For KTA-225
 * by Ocean Controls www.oceancontrols.com.au
 *
 * Controls 
 *  -8 Relays (4 With PWM Out (Should only be used with SSR's, the SSR Switching time is 1ms))
 *  -8 Analog Ins (or Non Isolated Digitals)
 */
#include <string.h> //Use the string Library
#include <ctype.h>
#include <EEPROM.h>
//void readdips(void);
void writedops(void);
void printaddr(char x);
void setbaud(char Mybaud);

#define ANIN1 0   // Analog 1 is connected to Arduino Analog In 0
#define ANIN2 1   // Analog 2 is connected to Arduino Analog In 1
#define ANIN3 2   // Analog 3 is connected to Arduino Analog In 2
#define ANIN4 3   // Analog 4 is connected to Arduino Analog In 3
#define ANIN5 4   // Analog 5 is connected to Arduino Analog In 4
#define ANIN6 5   // Analog 6 is connected to Arduino Analog In 5
#define ANIN7 6   // Analog 7 is connected to Arduino Analog In 6
#define ANIN8 7   // Analog 8 is connected to Arduino Analog In 7
int Analogs[8] = {ANIN1, ANIN2, ANIN3, ANIN4, ANIN5, ANIN6, ANIN7, ANIN8};

#define REL1 2  // Relay 1 is connected to Arduino Digital 2
#define REL2 3  // Relay 2 is connected to Arduino Digital 3 PWM
#define REL3 4  // Relay 3 is connected to Arduino Digital 4
#define REL4 5  // Relay 4 is connected to Arduino Digital 5 PWM
#define REL5 6  // Relay 5 is connected to Arduino Digital 6 PWM
#define REL6 7  // Relay 6 is connected to Arduino Digital 7
#define REL7 8  // Relay 7 is connected to Arduino Digital 8
#define REL8 9  // Relay 8 is connected to Arduino Digital 9 PWM
int Relays[8] = {REL1, REL2, REL3, REL4, REL5, REL6, REL7, REL8};
int PWMAble[4] = {2, 4, 5, 8};
/*
#define OI1 15 // Opto-Isolated Input 1 is connected to Arduino Analog 1 which is Digital 15
#define OI2 16 // Opto-Isolated Input 2 is connected to Arduino Analog 2 which is Digital 16
#define OI3 17 // Opto-Isolated Input 3 is connected to Arduino Analog 3 which is Digital 17
#define OI4 18 // Opto-Isolated Input 4 is connected to Arduino Analog 4 which is Digital 18
int Optos[4] = {OI1, OI2, OI3, OI4};
*/
#define TXEN 10 // RS-485 Transmit Enable is connected to Arduino Digital 10

char i = 0, Rxchar, Rxenable, Rxptr, Cmdcomplete, R;
char Rxbuf[15];
char adrbuf[3], cmdbuf[3], valbuf[12];
char rxaddress, Unitaddress, Unitbaud;
char Dip[5], Dop[9];
char Hold, Mask, Analog =1, Imax;
int val = 0, Param;      
int anreadings[3];
char x;
int  N, outval;
unsigned long RelayStartTime[8], WatchdogStartTime;
unsigned char RelayOnTime[8], WatchdogOnTime;

void setup() 
{
  analogReference(EXTERNAL); // use the AREF pin for the reference voltage so we can select between 5v or 3.3v
  for (i=0;i<8;i++)
  {
    pinMode(Relays[i], OUTPUT);  // declare the relay pin as an output
  }  
 /* for (i=0;i<4;i++)
  {
    pinMode(Optos[i], INPUT);  // declare the relay pin as an output
  } */
  pinMode(TXEN, OUTPUT);
  //Read Address, Baud and Parity from EEPROM Here  
  Unitaddress = EEPROM.read(0);
  if ((Unitaddress > 99) || (Unitaddress < 0))
  {
    Unitaddress = 0;  
  }
  Unitbaud = EEPROM.read(1);
//  Unitparity = EEPROM.read(2);//Parity currently not available
  setbaud(Unitbaud);// start serial port
   
  UCSR0A=UCSR0A |(1 << TXC0); //Clear Transmit Complete Flag
  digitalWrite(TXEN, HIGH);   //Enable Transmit
  delay(1);                   //Let 485 chip go into Transmit Mode
  Serial.println("Ocean Controls");
  Serial.println("KTA-225 v1.1");
  while (!(UCSR0A & (1 << TXC0)));    //Wait for Transmit to finish
  digitalWrite(TXEN,LOW);               //Turn off transmit enable

  //We want the switching freq to be pretty slow as the turn on time for the SSR's is 1ms
  //TCCR0B = TCCR0B & 0b11111101; // Arduino pins 5+6 61Hz 
  //TCCR1B = TCCR1B & 0b11111101; // Arduino pins 9+10 30Hz
  //TCCR2B = TCCR2B & 0b11111111; // Arduino pins 11+3 30Hz (stuck at 490Hz have to check this)
}

void loop() 
{
  //readdips();//read inputs
  //writedops();//set the relays
  if (Serial.available() > 0)    // Is a character waiting in the buffer?
  {
    Rxchar = Serial.read();      // Get the waiting character

    if (Rxchar == '@')      // Can start recording after @ symbol
    {
      if (Cmdcomplete != 1)
      {
        Rxenable = 1;
        Rxptr = 1;
      }//end cmdcomplete
    }//end rxchar
    if (Rxenable == 1)           // its enabled so record the characters
    {
      if ((Rxchar != 32) && (Rxchar != '@')) //dont save the spaces or @ symbol
      {
        Rxbuf[Rxptr] = Rxchar;
        //Serial.println(Rxchar);
        Rxptr++;
        if (Rxptr > 13) 
        {
          Rxenable = 0;
        }//end rxptr
      }//end rxchar
      if (Rxchar == 13) 
      {
        Rxenable = 0;
        Cmdcomplete = 1;
      }//end rxchar
    }//end rxenable

  }// end serial available


   //we should now have AACCXXXX in the rxbuf array, with a cr at rxptr
   //we should take rxbuf(1) and rxbuf(2) and turn them into a number

  if (Cmdcomplete == 1)
  {
     adrbuf[0] = Rxbuf[1];
     adrbuf[1] = Rxbuf[2];
     adrbuf[2] = 0; //null terminate Mystr = Chr(rxbuf(1)) + Chr(rxbuf(2))
     rxaddress = atoi(adrbuf);//    Address = Val(mystr)
  //Serial.println(adrbuf);
     cmdbuf[0] = toupper(Rxbuf[3]); //copy and convert to upper case
     cmdbuf[1] = toupper(Rxbuf[4]); //copy and convert to upper case
     cmdbuf[2] = 0; //null terminate        Command = Chr(rxbuf(3)) + Chr(rxbuf(4))
     //   Command = Ucase(command)
  //Serial.println(cmdbuf);
     valbuf[0] = Rxbuf[5]; //        Mystr = Chr(rxbuf(5))
        R = Rxptr - 1;
            for (i = 6 ; i <= R ; i++)//For I = 6 To R
            {
                valbuf[i-5] = Rxbuf[i]; //Mystr = Mystr + Chr(rxbuf(i))
            }
     valbuf[R+1] = 0; //null terminate
     Param = atoi(valbuf);//   Param = Val(mystr)

     //Serial.println(Param); //   'Print "Parameter: " ; Param

       if ((rxaddress == Unitaddress) || (rxaddress == 0)) //0 is wildcard address, all units respond
       {

         //switch (cmdbuf) //Select Case Command
         //{
              if (strcmp(cmdbuf,"ON")==0)                                   //'turn relay x ON
              {
                   //'Print "command was ON"
                   if ((Param <= 8) && (Param >= 0)) 
                   {
                     if (Param == 0)
                     { 
                        for (i = 1 ; i<=8 ; i++)
                        {
                          Dop[i-1] = 1;
                          RelayOnTime[i-1]= 0;//override the timing
                        }
                     }
                     else
                     {
                        Dop[Param-1] = 1;
                        RelayOnTime[Param-1]= 0;//override the timing
                     }
                     writedops();
                     printaddr(1);                     
                   }
                   else
                   {
                        //'Print "out of range"
                   }
              }
              if (strcmp(cmdbuf,"OF")==0)                                   //'turn relay x OFF
              {
                   //'Print "command was OFF"
                   if ((Param <= 8) && (Param >= 0)) 
                   {
                     if (Param == 0)
                     {
                        for (i = 1 ; i<=8 ; i++)
                        {
                          Dop[i-1] = 0;
                        }
                     }
                     else
                     {
                        Dop[Param-1] = 0;
                     }
                     writedops();
                     printaddr(1);
                   }
                   else
                   {
                        //'Print "out of range"
                   }
              }    
              /*
              if (strcmp(cmdbuf,"PW")==0)        // PWM out on Relays 2, 4, 5 or 8 only
              {                                  // Value of parameter is Relay number and 0-255 value ie 2127 will make channel 2 output 127 (50%)
                   for ( i=0 ; i<=3 ; i++)       //
                   {
                     x = Param / 1000;
                     if ( x==PWMAble[i]) //Is is on the list?
                     {
                       
                       outval = Param % 1000; //Get the 0-255 value
                       if (outval>=0 && outval <=255) //within range?
                       {
                         analogWrite( Relays[x-1] , outval);
                         if (outval == 0)
                         {
                           Dop[x-1]=0;     //If it is 0 then we will say the Relay is OFF
                         }
                         else
                         {
                           Dop[x-1]=1;     //If it is not 0 then we will say the Relay is ON
                         }
                         printaddr(1);
                       }
                     }
                   }
              }
              */
              if (strcmp(cmdbuf,"TR")==0)        // Timed Relay
              {                                  // Value of parameter is Relay number and 1-255 time in 0.1s increments 2127 is relay 2 for 12.7 sec
                 x = Param / 1000; //Get the relay number
                 outval = Param % 1000; //Get the 0-255 value
                 if (outval>=1 && outval <=255) //within range?
                 {
                     RelayStartTime[x-1] = millis(); //save the current time
                     RelayOnTime[x-1] = outval;
                     Dop[x-1]=1;     //Relay on 
                     writedops();    //Write outputs
                     printaddr(1);
                 }
              }
              
              if (strcmp(cmdbuf,"KA")==0)        // Keep Alive
              {                                  // Value of parameter is seconds to stay alive for
                 
                 if (Param>=0 && Param <=255) //within range?
                 {
                     WatchdogStartTime = millis(); //save the current time
                     WatchdogOnTime = Param;
                     printaddr(1);
                 }
              }
              
              if (strcmp(cmdbuf,"WR")==0)                                    //'turn relay on/off according to binary value of x
              {
                    //'Print "command was WRITE"
                   if ((Param <= 255) && (Param >= 0)) 
                   {
                    for (i=0 ; i<8 ; i++)
                    {
                       Mask = 1<<i;//Shift Mask , Left , I
                       Hold = Param & Mask;
                       /*'Print "mask=" ; Mask
'                                Print " I= " ; I
'                                Print "param=" ; Param
'                                print "hold=" ; Hold
*/
                       if (Hold == Mask)
                       {
                          Dop[i] = 1;
                          RelayOnTime[i]= 0; //override timing
                       }
                       else
                       {
                          Dop[i] = 0;
                       }
                    }
                   }
                   else
                   {
                   }
                   writedops();
                   printaddr(1);
                   
              }
              
              if (strcmp(cmdbuf,"RS")==0)                                   // 'relay status
              {
                    //'Print "command was RELAY STATUS"
                    if ((Param > 0) && (Param <= 8))
                   {
                         printaddr(2);
                         Serial.println(Dop[Param-1], DEC);
                   }
                   else if (Param == 0) 
                   {
                         N = 0;
                         for (i=0 ; i<8 ; i++)
                         {
                            if(Dop[i] == 1)
                            {
                               N = N|(1<<i);
                            }
                         }
                         printaddr(2);
                         Serial.println(N, DEC);
                   }
                   else
                   {
                     //'Print "out of range"
                   }
              }   

              /*if (strcmp(cmdbuf,"IS")==0)                                   // 'input status
              {
                readdips();                 
                     if ((Param > 0) && (Param <= 4)) 
                     {
                       printaddr(2);
                       Serial.println(Dip[Param-1], DEC);
                       
                     }
                     else if (Param == 0)
                     {
                        N = 0;
                        if (Analog == 1)
                        {
                          Imax = 3;
                        }
                        else 
                        {
                          Imax = 8;
                        }

                        for (i=0 ; i<=Imax ; i++)
                        {
                          if (Dip[i] == 1)
                          {
                           N = N|(1<<i);
                          }
                        }
                        printaddr(2);
                        Serial.println(N,DEC);
          
                     }
                     else
                     {
                      //        'Print "out of range"
                     }
              }
              */

              if (strcmp(cmdbuf,"AI")==0)                                   // 'return analog input
              {                                   
                    if (Analog == 1)
                    {
                       if ((Param >= 0) && (Param <= 8))
                       {
                         for (i=0 ; i<8 ; i++)
                         {
                           anreadings[i] = analogRead(Analogs[i]);
                         }
                         printaddr(2);
                         if (Param == 0)
                         {
                             for (i=0 ; i<7 ; i++)
                             {
                               Serial.print(anreadings[i], DEC);
                               Serial.print(" ");
                             }
                             
                             Serial.println(anreadings[7], DEC);
                             
                         }
                         else
                         {
                             Serial.println(anreadings[Param-1], DEC);
                         }
                       }
                    }
                    else
                    {
                    }
              }
              if (strcmp(cmdbuf,"SS")==0)                                   // System Status
              {
                    
                   if (Param == 0)
                   {
                         N = 0;
                         for (i=0 ; i<8 ; i++) //Read Relays
                         {
                            if(Dop[i] == 1)
                            {
                               N = N|(1<<i);
                            }
                         }
                         printaddr(2);
                         Serial.print(N, DEC); //Print Relays
                         Serial.print(" ");
                         /*
                          readdips(); //Read Inputs
                          N = 0;
                          if (Analog == 1)
                          {
                            Imax = 3;
                          }
                          else 
                          {
                            Imax = 8;
                          }
  
                          for (i=0 ; i<=Imax ; i++)
                          {
                            if (Dip[i] == 1)
                            {
                             N = N|(1<<i);
                            }
                          }
                         
                          Serial.print(N,DEC); //Print Inputs
                          Serial.print(" ");
                          */
                          if (Analog == 1)
                          {
                             for (i=0 ; i<8 ; i++)
                             {
                               anreadings[i] = analogRead(Analogs[i]); // Read Analogs
                             }
                                 for (i=0 ; i<7 ; i++)
                                 {
                                   Serial.print(anreadings[i], DEC);
                                   Serial.print(" ");
                                 }
                                 Serial.println(anreadings[7], DEC);
                                 
                          }
                   }
                   else
                   {
                     //'Print "out of range"
                   }
              }   
              if (strcmp(cmdbuf,"SA")==0)                                   // Set Address and save to EEP
              {
                    //'Print "command was Set Address"
                   if ((Param >= 0) && (Param <= 99))
                   {
                     Unitaddress = Param;   //make it the address
                     EEPROM.write(0, Unitaddress);//save to eep                       
                     printaddr(1);                         
                   }
                   else
                   {
                     //'Print "out of range"
                   }
              } 
              if (strcmp(cmdbuf,"SB")==0)                                   // Set Baud and save to EEP
              {
                    //'Print "command was Set Baud"
                   if ((Param > 0) && (Param <= 10))
                   {
                     Unitbaud = Param;   
                     EEPROM.write(1, Unitbaud);//save to eep          

                     setbaud(Unitbaud);// start serial port             
                     printaddr(1);                         
                   }
                   else
                   {
                     //'Print "out of range"
                   }
              } 
/*
             case "SB":                                     'set Baud
                   If Param > 0 And Param < 13 Then
                     Mybaud = Param
                     Save_eep
                     If Serdef = False Then Setbaud
                     If unitAddress < 10 Then
                         Print "#0" ; unitAddress
                     Else
                         Print "#" ; unitAddress
                     End If
                   End If

             case "SP":                                     'set parity
                   If Param >= 0 And Param < 4 Then
                     Myparity = Param
                     Save_eep
                     if serdef = false then Setparity
                     If unitAddress < 10 Then
                         Print "#0" ; unitAddress
                     Else
                         Print "#" ; unitAddress
                     End If
                   End If
*/                   
/*'             case "SS":                                     'set stop bits
'                Mystopbits = Param
'                if serdef = false then Setstopbits

'             case "SD":                                     'set data bits
'                Mydatabits = Param
'                if serdef = false then Setdatabits
*/

         //}//end switch cmdbuf
       }//end address


      Cmdcomplete = 0;
  }//end cmdcomplete 
    while (!(UCSR0A & (1 << TXC0)));    //Wait for Transmit to finish
    digitalWrite(TXEN,LOW);               //Turn off transmit enable
    checkTime();
}//end loop

void checkTime(void)
{
  unsigned long currentTime = millis();
  for (i=0 ; i<8 ; i++)
  {
    if (RelayOnTime[i] != 0)
    {
      if (currentTime - RelayStartTime[i] >= (unsigned long)(RelayOnTime[i]) * 100) //Check if time to turn off relays
      {
          RelayOnTime[i]= 0;
          Dop[i]= 0; //turn off relay
          writedops(); //write outputs
      }
    }
  }
  if (WatchdogOnTime != 0)
  {
    if (currentTime - WatchdogStartTime >= (unsigned long)(WatchdogOnTime) * 1000) //Watchdog timed out
    {
      for (i=0 ; i<8 ; i++)
      {
        Dop[i] = 0; //turn off relay
      }
      writedops(); //write outputs
      WatchdogStartTime = currentTime; //keep the same timeout until it is turned off
    }
  }
}

/*void readdips(void)
{
  
  for (i=0 ; i<4 ; i++)
  {
    if (digitalRead(Optos[i])==LOW)
    {
      Dip[i] = 1;
    }
    else
    {
      Dip[i] = 0;
    }
  }
}
*/
void writedops(void)
{
  for (i=0 ; i<8 ; i++)
  {
    if (Dop[i]==1)
    {
      digitalWrite(Relays[i],HIGH);
    }
    else
    {
      digitalWrite(Relays[i],LOW);
    }
  }
}

void printaddr(char x) //if x=1 then it prints an enter, if x=2 then it prints a space after the address
{
  UCSR0A=UCSR0A |(1 << TXC0); //Clear Transmit Complete Flag
  digitalWrite(TXEN, HIGH);   //Enable Transmit
  delay(1);                   //Let 485 chip go into Transmit Mode
  
  if (Unitaddress < 10)
  {
    Serial.print("#0");
    Serial.print(Unitaddress, DEC);
  }
  else
  {
    Serial.print("#"); 
    Serial.print(Unitaddress, DEC);
  }
  switch(x)
  {
    case 1:
        Serial.println(); //print enter
      break;
    case 2:
        Serial.print(" "); //print space
      break;
  
  }
}

void setbaud(char Mybaud)
{
   switch (Mybaud)
   {
    case 1 : Serial.begin(1200);
      break;
    case 2 : Serial.begin(2400);
      break;     
    case 3 : Serial.begin(4800);
      break;
    case 4 : Serial.begin(9600);
      break;
    case 5 : Serial.begin(14400);
      break;
    case 6 : Serial.begin(19200);
      break;
    case 7 : Serial.begin(28800);
      break;
    case 8 : Serial.begin(38400);
      break;
    case 9 : Serial.begin(57600);
      break;
    case 10 : Serial.begin(115200);
      break;
    default:  Serial.begin(9600);
      break;
   }
}
