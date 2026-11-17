# ESP32 IoT Physical Buzzer Integration Guide

This document describes the physical IoT buzzer contract and provides an Arduino sketch for connecting an ESP32 microcontroller to the ForestGuard AI alerting subsystem.

> **Status in current release:** Modular Stub (Disabled by default). Hardware is not required to run the demo.

---

## 1. Webhook Contract

When an emergency threat is confirmed and `ESP32_BUZZER_URL` is configured, the backend dispatches an HTTP GET request to the ESP32:

```http
GET http://<esp32-ip>/buzz?ms=3000
```

- **Parameters**:
  - `ms` (optional): Duration in milliseconds to sound the buzzer (default: 3000 ms).
- **Expected Response**:
  ```json
  {"status": "buzzing", "duration_ms": 3000}
  ```

---

## 2. Example Arduino / ESP32 Sketch

```cpp
#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

const int BUZZER_PIN = 18;  // GPIO pin connected to active buzzer or relay
WebServer server(80);

void handleBuzz() {
  int duration = 3000;
  if (server.hasArg("ms")) {
    duration = server.arg("ms").toInt();
  }

  server.send(200, "application/json", "{\"status\":\"buzzing\",\"duration_ms\":" + String(duration) + "}");

  // Sound active buzzer
  digitalWrite(BUZZER_PIN, HIGH);
  delay(duration);
  digitalWrite(BUZZER_PIN, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.print("ESP32 Connected! IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/buzz", HTTP_GET, handleBuzz);
  server.begin();
}

void loop() {
  server.handleClient();
}
```

---

## 3. How to Enable in ForestGuard AI

1. Flash the sketch to your ESP32 board and note its local IP address (e.g. `192.168.1.150`).
2. Add the URL to `backend/.env`:
   ```env
   ESP32_BUZZER_URL=http://192.168.1.150/buzz?ms=3000
   ```
3. In `backend/services/notifiers/esp32_buzzer.py`, set:
   ```python
   self.enabled = True
   ```
4. Restart the backend. Any confirmed fire detection will now physically buzz the remote watchtower alert siren.
