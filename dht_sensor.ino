#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);
int count = 0;

void setup() {
  Serial.begin(9600);
  dht.begin();
  delay(2000);
}

void loop() {
  delay(5000);
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    Serial.println("{\"error\":\"Sensor read failed\"}");
    return;
  }

  count++;
  Serial.print("{\"temp\":");
  Serial.print(t, 1);
  Serial.print(",\"humidity\":");
  Serial.print(h, 1);
  Serial.print(",\"count\":");
  Serial.print(count);
  Serial.println("}");
}
