# Screen Monitor - Installation and Setup

## Requirements (requirements.txt)
```
pillow>=9.0.0
opencv-python>=4.5.0
pytesseract>=0.3.10
pynput>=1.7.6
pyaudio>=0.2.11
psutil>=5.9.0
numpy>=1.21.0
```

## System Dependencies

### Windows:
1. Install Tesseract OCR:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or set TESSDATA_PREFIX environment variable

2. Install Visual C++ Build Tools (if needed for PyAudio)

### macOS:
```bash
brew install tesseract
brew install portaudio  # for PyAudio
```

### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
sudo apt-get install portaudio19-dev  # for PyAudio
sudo apt-get install python3-tk  # for PIL
```

## Installation Steps

1. Create virtual environment:
```bash
python -m venv screen_monitor_env
source screen_monitor_env/bin/activate  # Linux/Mac
# or
screen_monitor_env\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test Tesseract installation:
```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

## Quick Start

1. Run the basic implementation:
```bash
python screen_monitor.py
```

2. The program will:
   - Create a 'screen_data' directory
   - Start capturing screenshots every 10 seconds
   - Monitor mouse clicks and keyboard presses
   - Extract text from screenshots using OCR
   - Store all data in SQLite database

3. Stop with Ctrl+C

## Output Structure
```
screen_data/
├── screenshots/          # PNG files of screen captures
├── audio/               # WAV files (if audio enabled)
├── screen_activity.db   # SQLite database with all events
├── screen_monitor.log   # Log file
└── report.json         # Summary report
```

## Database Schema

### Screenshots Table:
- id, timestamp, filename, text_content, window_title, active_app

### Mouse Events Table:
- id, timestamp, event_type, x, y, button

### Keyboard Events Table:
- id, timestamp, event_type, key, text

### System Events Table:
- id, timestamp, event_type, data

## Security and Privacy Notes

⚠️ **IMPORTANT**: This tool captures sensitive information including:
- Screenshots of your screen
- Keyboard input (passwords, personal data)
- Mouse activity
- Audio (if enabled)

**Recommendations**:
- Only use on your own devices
- Encrypt the output directory
- Regularly clean old data
- Be aware of legal requirements in your jurisdiction
- Consider implementing data retention policies

## Troubleshooting

### Common Issues:

1. **Tesseract not found**:
   - Ensure Tesseract is installed and in PATH
   - Set pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\tesseract.exe'

2. **PyAudio installation fails**:
   - Windows: Install Visual C++ Build Tools
   - macOS: Install portaudio with brew
   - Linux: Install portaudio19-dev

3. **Permission errors**:
   - Run with appropriate permissions
   - Some systems require special permissions for input monitoring

4. **High CPU usage**:
   - Increase screenshot_interval
   - Disable OCR processing
   - Reduce image resolution before processing