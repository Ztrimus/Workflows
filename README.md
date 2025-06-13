# Screen Monitor - Installation and Setup

## Permission Need
- For Mac: Give permission "Screen & System Audio Recording" to VS Code 
- brew install tesseract

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



"""
## Complete Workflow Analysis and Automation System

We have implemented a comprehensive three-stage pipeline for screen activity analysis and workflow automation:

### Stage 1: Data Collection ✅
Implemented in `main.py` - Collects screen activity data into a unified database including:
- Screenshots with OCR text extraction
- Mouse events (clicks, movements)
- Keyboard events (key presses)
- System information (active apps, window titles, clipboard)
- Multi-monitor support
- Timestamp-based event tracking

### Stage 2: Workflow Analysis ✅
Implemented in `workflow_analyzer.py` - Analyzes collected data to understand user workflows:

**Key Features:**
1. **Session Identification**: Groups activities into logical work sessions
2. **Application Workflow Extraction**: Identifies workflows per application
3. **Pattern Detection**: Finds repetitive user behaviors
4. **Documentation Generation**: Creates human-readable workflow documentation
5. **Visualization Creation**: Generates charts, heatmaps, and network graphs
6. **Knowledge Graph**: Creates visual representations of workflow relationships

**Analysis Outputs:**
- Workflow documentation (Markdown format)
- Activity visualizations (timeline, heatmap, network graphs)
- Keyword clouds from screen content
- Session reports with statistics
- Repetitive pattern identification

### Stage 3: Workflow Automation ✅
Implemented in `workflow_replayer.py` - Enables workflow replication and optimization:

**Key Features:**
1. **Workflow Replay**: Automatically execute recorded workflows
2. **Noise Reduction**: Filter out irrelevant actions
3. **Workflow Optimization**: Remove redundant steps
4. **Dry Run Mode**: Test workflows without execution
5. **Safety Controls**: Built-in safeguards and user confirmations
6. **Speed Control**: Adjustable execution speed

**Automation Capabilities:**
- Cross-platform application launching
- Mouse click automation
- Keyboard input simulation
- Workflow complexity analysis
- Optimization recommendations
- Automated script generation

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Collection   │ -> │  Workflow Analysis  │ -> │   Automation    │
│    (main.py)       │    │ (workflow_analyzer) │    │  (replayer.py)  │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Screenshots   │    │ • Session Detection │    │ • Workflow Replay│
│ • Mouse Events  │    │ • Pattern Analysis  │    │ • Optimization  │
│ • Keyboard      │    │ • Documentation     │    │ • Safety Controls│
│ • System Info   │    │ • Visualizations    │    │ • Script Gen    │
│ • OCR Text      │    │ • Knowledge Graphs  │    │ • Cross-platform│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Use Cases Addressed

1. **Automatic Documentation**: 
   - Captures what work was done
   - Generates step-by-step documentation
   - Creates visual workflow representations
   - Maintains knowledge base of completed tasks

2. **Task Replication**:
   - Records complex workflows
   - Identifies optimal steps
   - Removes noise and redundancy
   - Enables one-click task automation

3. **Workflow Optimization**:
   - Analyzes user behavior patterns
   - Identifies inefficiencies
   - Suggests improvements
   - Creates optimized automation scripts
"""

## References
- https://github.com/quillcraftsman/replay-wizard
- https://replaywizard.craftsman.lol/
- [How to use Microsoft Power Automate Desktop - Full tutorial](https://www.youtube.com/watch?v=IQ_KpBC8fwo)
- https://github.com/alyvix/alyvix

## Papers
- https://github.com/niuzaisheng/ScreenAgent
- https://github.com/niuzaisheng/ScreenAgentWebClient
- https://arxiv.org/abs/2108.04212

## Articles
- [Workflow Analysis Explained – Methods, Steps & Tools](https://www.businessprocessincubator.com/content/workflow-analysis-explained-methods-steps-tools/)
- [The Ultimate Guide to Workflow Analysis](https://thedigitalprojectmanager.com/topics/best-practices/workflow-analysis/)

## Useful things
- pyautogui.alert: to notify user
- pyautogui.prompt: to ask user for input for human in loop
- pyautogui.write(text, interval): to write text with speed interval
- pyautogui.click(x, y): to click at x, y
- pyautogui.moveTo(x, y): to move mouse to x, y
- pyautogui.screenshot(): to take screenshot
- https://digitalworkflow.io/python-pyautogui-useful-scripts/
- https://codezup.com/automating-tasks-with-python-and-the-pyautogui-library/

# Next TODO
- [ ] Explore alternative OCR libraries like DocTR or EasyOCR for potentially better accuracy than Tesseract
- [ ] Study the maCrow repository for JSON-based action storage patterns that might improve your data structure
- [ ] Integrate a robust screen capture layer: Choose between DXcam (Windows), python-mss (cross‑platform), or D3DShot.
- [ ] Explore Recorder + Screener modules: Review Alyvix and Screenshot_LLM code for GUI detection and session handling ideas.
- [ ] Study Session Replay & LLM agents: ScreenAgent and Mixpanel/Datadog docs can guide action-extraction and automation script generation.
- [ ] Deepen pipeline architecture understanding: Workflow engines video and blog tutorials show scalable modular design approaches.