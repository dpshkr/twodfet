//#include <OPAMP.h>

/* TIMING PARAMETERS */
const unsigned int pulseWidth = 10 ; // Transistor turn on pulse width in ms
const unsigned int integrationTime = 5000; // Integration time in ms

/* PIN VALUES */
const int transistorControlPin = 7; // Pin connected to transistor
const int outPin = A0;


unsigned long previousTime = 0;
unsigned long currentTime;
int outValue;

void setup() {
  // put your setup code here, to run once:
  //analogReadResolution(14);
  //OPAMP.begin(OPAMP_SPEED_LOWSPEED);
  pinMode(transistorControlPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {

  outValue = analogRead(outPin);
  Serial.println(outValue);

  currentTime = millis();

  if (currentTime - previousTime >= integrationTime) {
    digitalWrite(transistorControlPin, HIGH);
    previousTime = currentTime;
    while (millis() - currentTime <= pulseWidth) {
      outValue = analogRead(outPin);
      Serial.println(outValue);
    }
    digitalWrite(transistorControlPin, LOW);
  }
}
