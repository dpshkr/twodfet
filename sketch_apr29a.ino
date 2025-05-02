int sensorValue;
unsigned long thisTime, initTime;
unsigned long integrationTime = 1000;

void setup() {
  // put your setup code here, to run once:
  initTime = millis();
  Serial.begin(9600);
  pinMode(7, OUTPUT);

}

void loop() {
  sensorValue = analogRead(A0);
  Serial.println(sensorValue);
  // put your main code here, to run repeatedly:
  thisTime = millis() - initTime;
  if (thisTime % integrationTime < 50) {
    digitalWrite(7, HIGH);
    delay(10);
    digitalWrite(7, LOW);
  }
}
