#include <Adafruit_AS7341.h>
#include <Adafruit_SHT4x.h>
#include <Wire.h>

// Initialize SHT4x sensor (temperature & humidity)
Adafruit_SHT4x sht4 = Adafruit_SHT4x();

// Initialize two AS7341 spectral sensors
Adafruit_AS7341 as7341_1;
Adafruit_AS7341 as7341_2;

// Setup function: runs once at startup
void setup() {
  Serial.begin(115200);

  // Wait for serial port to connect
  while (!Serial) {
    delay(1);
  }

  // Set pin 2 as output (used for LED indicator)
  pinMode(2, OUTPUT);

  /// Initialize AS7341 sensor 1
  //Wire.begin(/*SDA*/15,/*SCL*/4);

  if (!as7341_1.begin()){
    Serial.println("Could not find as7341_1");
    while (1) { delay(10); }
  }

  Serial.println("Found as7341_1 sensor");

  // Configure integration time and gain
  as7341_1.setATIME(100);
  as7341_1.setASTEP(999);
  as7341_1.setGain(AS7341_GAIN_256X);

  /// Initialize SHT4x sensor
  //Wire1.begin(/*SDA*/21,/*SCL*/22);

  if (! sht4.begin()) {
    Serial.println("Couldn't find SHT4x");
    while (1) delay(1);
  }

  Serial.println("Found SHT4x sensor");

  // Print sensor serial number
  Serial.print("Serial number 0x");
  Serial.println(sht4.readSerial(), HEX);

  // Set measurement precision (higher = more accurate but slower)
  sht4.setPrecision(SHT4X_HIGH_PRECISION);

  switch (sht4.getPrecision()) {
    case SHT4X_HIGH_PRECISION:
      Serial.println("High precision");
      break;
    case SHT4X_MED_PRECISION:
      Serial.println("Med precision");
      break;
    case SHT4X_LOW_PRECISION:
      Serial.println("Low precision");
      break;
  }

  // Configure heater (disabled here to avoid affecting measurements)
  sht4.setHeater(SHT4X_NO_HEATER);

  switch (sht4.getHeater()) {
    case SHT4X_NO_HEATER:
      Serial.println("No heater");
      break;
    case SHT4X_HIGH_HEATER_1S:
      Serial.println("High heat for 1 second");
      break;
    case SHT4X_HIGH_HEATER_100MS:
      Serial.println("High heat for 0.1 second");
      break;
    case SHT4X_MED_HEATER_1S:
      Serial.println("Medium heat for 1 second");
      break;
    case SHT4X_MED_HEATER_100MS:
      Serial.println("Medium heat for 0.1 second");
      break;
    case SHT4X_LOW_HEATER_1S:
      Serial.println("Low heat for 1 second");
      break;
    case SHT4X_LOW_HEATER_100MS:
      Serial.println("Low heat for 0.1 second");
      break;
  }
}

// Main loop: runs repeatedly
void loop() {

  // Turn LED ON to indicate measurement start
  digitalWrite(2, HIGH);

  // Arrays to store raw and processed spectral data
  uint16_t readings_1[12];
  uint16_t readings_2[12];
  float counts_1[12];
  float counts_2[12];

  // Variables for temperature and humidity data
  sensors_event_t humidity, temp;

  // Timestamp for measuring sensor read duration
  uint32_t timestamp = millis();

  //// Read AS7341 spectral data
  if (!as7341_1.readAllChannels(readings_1)){
    Serial.println("As7341_1 error reading all channels!");
    return;
  }

  // Convert raw readings to basic counts
  for(uint8_t i = 0; i < 12; i++) {
    // Skip duplicate Clear/NIR channels (index 4 and 5)
    if(i == 4 || i == 5) continue;

    counts_1[i] = as7341_1.toBasicCounts(readings_1[i]);
  }

  // Print spectral channel data
  Serial.print("As7341_1 F1 415nm : ");
  Serial.println(counts_1[0]);

  Serial.print("As7341_1 F2 445nm : ");
  Serial.println(counts_1[1]);

  Serial.print("As7341_1 F3 480nm : ");
  Serial.println(counts_1[2]);

  Serial.print("As7341_1 F4 515nm : ");
  Serial.println(counts_1[3]);

  Serial.print("As7341_1 F5 555nm : ");
  Serial.println(counts_1[6]);  // skipping duplicate indices

  Serial.print("As7341_1 F6 590nm : ");
  Serial.println(counts_1[7]);

  Serial.print("As7341_1 F7 630nm : ");
  Serial.println(counts_1[8]);

  Serial.print("As7341_1 F8 680nm : ");
  Serial.println(counts_1[9]);

  // Print Clear and NIR channels
  Serial.print("As7341_1 Clear    : ");
  Serial.println(counts_1[10]);

  Serial.print("As7341_1 NIR      : ");
  Serial.println(counts_1[11]);

  Serial.println();

  //// Read SHT4x temperature & humidity
  sht4.getEvent(&humidity, &temp); // update readings

  timestamp = millis() - timestamp;
 // Print temperature and humidity
  Serial.print("Sht41 Temperature: ");
  Serial.print(temp.temperature);
  Serial.println(" degrees C");

  Serial.print("Sht41 Humidity: ");
  Serial.print(humidity.relative_humidity);
  Serial.println("% rH");

  Serial.print("Sht41 Read duration (ms): ");
  Serial.println(timestamp);

  Serial.println();

  // Delay between measurements
  delay(500);

  // Turn LED OFF after measurement
  digitalWrite(2, LOW);

  delay(500);
}