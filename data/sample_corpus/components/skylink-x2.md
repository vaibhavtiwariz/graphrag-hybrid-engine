# SkyLink X2 Radio Module

The SkyLink X2 is a dual-band 2.4/5 GHz telemetry radio module manufactured by BlueFin Circuits.
It is used in the NimbusOne drone and in HaloDock dock controllers.

## Firmware history

- 2.3.0 (February 2024): initial production firmware.
- 2.3.1 (April 2024): introduced a watchdog reset defect causing intermittent link drops under
  sustained vibration; implicated in fleet incident INC-1077.
- 2.3.2 (June 2024): fixes the watchdog reset bug and ships cryptographically signed images,
  satisfying the SEC-14 signed-firmware security standard.

Firmware is distributed through the secure OTA channel only.
