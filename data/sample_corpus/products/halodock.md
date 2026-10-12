# HaloDock Wireless Charging Dock

HaloDock is a wireless charging dock for the NimbusOne drone, released in January 2025 with a
900 W output. Approximately 300 depots operate HaloDock stations.

## Key subsystems

- Power relay: a 20 A relay supplied by BlueFin Circuits (revision A).
- Enclosure: fabricated by Kestrel Metals.
- Load balancing: requires dock firmware 3.1 or later to stagger simultaneous charge sessions.

## Known constraints

Early revision-A relays can trip depot breakers under full 900 W simultaneous load; see incident
INC-1090 for remediation status. Dock controllers fall under the signed-firmware rule SEC-14.
