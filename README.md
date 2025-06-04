# References
- https://arxiv.org/abs/2410.18963
- https://github.com/niuzaisheng/ScreenAgent?tab=readme-ov-file
- https://arxiv.org/pdf/2410.18963
- https://arxiv.org/pdf/2312.13108
- Power Automate: https://learn.microsoft.com/en-us/power-automate/desktop-flows/introduction


# Todo
- [ ] pyautogui
- [ ] HumanInput libraries javascript
- [ ] pynput
- [ ] Application focus tracking
    - Windows: Query GetForegroundWindow() API
    - macOS: Use NSWorkspace notifications
- [ ] Audio Capture
    - sounddevice - python
    - pyaudio - python
- [ ] Real-Time Processing Pipeline
Data Sources → Stream Processor → Storage → Analytics
       ↑           ↓          ↓
    (Screen)  (Apache Kafka) (Redis)
       ↑           ↓
   (Mouse/Key) (Apache Flink)
