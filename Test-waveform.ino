#include <Arduino.h>
 
const int NUM_SENSORS = 8;
const int sensorPins[NUM_SENSORS] = {34, 35, 32, 33, 25, 26, 27, 12};
 
int baseline[NUM_SENSORS];
 
void setup() {
  Serial.begin(115200);
  delay(1000);
 
  btStop();
 
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
 
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(sensorPins[i], INPUT);
  }
 
  long sums[NUM_SENSORS] = {0};
  const int N = 200;
  for (int n = 0; n < N; n++) {
    for (int i = 0; i < NUM_SENSORS; i++) {
      sums[i] += analogRead(sensorPins[i]);
    }
    delay(2);
  }
  for (int i = 0; i < NUM_SENSORS; i++) {
    baseline[i] = sums[i] / N;
  }
 
  Serial.println("START");
  Serial.println("s1,s2,s3,s4,s5,s6,s7,s8");
}
 
void loop() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    int raw = analogRead(sensorPins[i]);
    int delta = abs(raw - baseline[i]);
    Serial.print(delta);
 
    if (i < NUM_SENSORS - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
 
  delay(50);
}