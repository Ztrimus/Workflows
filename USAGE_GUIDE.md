# Workflow Analysis and Automation System - Usage Guide

This guide explains how to use the complete workflow analysis and automation system to capture, analyze, and automate your computer tasks.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For macOS users, you may also need:
brew install tesseract
```

### 2. Basic Workflow

1. **Record your activities** using `main.py`
2. **Analyze the data** using `workflow_analyzer.py`
3. **Replay workflows** using `workflow_replayer.py`

## Detailed Usage

### Stage 1: Data Collection

#### Starting Screen Recording

```bash
python main.py
```

**What it does:**
- Captures screenshots every 5 seconds
- Records all mouse clicks and movements
- Logs keyboard inputs
- Extracts text from screen using OCR
- Tracks active applications and windows
- Monitors clipboard content

**Controls:**
- A system tray icon (📹) appears in your menu bar
- Click "Stop Recording" to end the session
- All data is saved to `screen_data/screen_activity.db`

**Configuration Options:**
Edit these settings in `main.py`:
```python
monitor.config["screenshot_interval"] = 5.0  # seconds between screenshots
monitor.config["ocr_enabled"] = True         # enable text extraction
monitor.config["save_screenshots"] = True    # save screenshot files
monitor.capture_audio = False                # audio recording (disabled)
```

### Stage 2: Workflow Analysis

#### Running Analysis

```bash
python workflow_analyzer.py
```

**What it analyzes:**
- **Sessions**: Groups activities by time gaps (default: 30 min idle = new session)
- **Application Workflows**: Identifies task sequences per application
- **Repetitive Patterns**: Finds recurring action sequences
- **Screen Content**: Extracts keywords and topics from OCR text
- **Activity Patterns**: Creates timeline and usage heatmaps

**Generated Outputs:**

1. **Reports** (`analysis_output/reports/`):
   - `workflow_analysis_results.json` - Complete analysis data

2. **Documentation** (`analysis_output/documentation/`):
   - `{AppName}_workflow.md` - Human-readable workflow docs

3. **Visualizations** (`analysis_output/visualizations/`):
   - `timeline.html` - Interactive application usage timeline
   - `workflow_network.png` - Network graph of workflow steps
   - `activity_heatmap.png` - Activity patterns by day/hour
   - `keyword_cloud.png` - Word cloud of screen content

4. **Automation Scripts** (`analysis_output/workflows/`):
   - `{AppName}_automation.py` - Generated automation scripts

#### Customizing Analysis

```python
# In workflow_analyzer.py, modify these parameters:
analyzer = WorkflowAnalyzer()

# Change session detection threshold
sessions = analyzer.identify_sessions(idle_threshold_minutes=45)

# Adjust workflow segmentation
# Edit _segment_app_workflow() gap_threshold_minutes parameter
```

### Stage 3: Workflow Automation

#### Interactive Workflow Replay

```bash
python workflow_replayer.py
```

**Interactive Menu:**
1. Lists all available workflows
2. Shows workflow details (duration, events, description)
3. Provides options for each workflow:
   - **Dry Run**: Simulate without executing
   - **Execute**: Run the actual workflow
   - **Optimize**: Create noise-reduced version
   - **Report**: Generate detailed analysis

#### Programmatic Usage

```python
from workflow_replayer import WorkflowReplayer

replayer = WorkflowReplayer()

# List available workflows
workflows = replayer.list_available_workflows()
print(f"Found {len(workflows)} workflows")

# Replay a specific workflow
workflow_id = "TextEdit_0"  # Example workflow ID
success = replayer.replay_workflow(workflow_id, speed_multiplier=1.5)

# Create optimized version
optimized = replayer.create_optimized_workflow(workflow_id)
print(f"Reduced events by {optimized['reduction_percentage']:.1f}%")
```

## Advanced Features

### Custom Analysis

#### Analyzing Specific Time Periods

```python
from workflow_analyzer import WorkflowAnalyzer
import pandas as pd

analyzer = WorkflowAnalyzer()

# Filter data by date range
start_date = "2024-01-01"
end_date = "2024-01-31"
filtered_df = analyzer.df[
    (analyzer.df['timestamp'] >= start_date) & 
    (analyzer.df['timestamp'] <= end_date)
]

# Analyze specific applications
app_data = analyzer.df[analyzer.df['active_app'] == 'Visual Studio Code']
```

#### Custom Pattern Detection

```python
# Find patterns with custom parameters
patterns = analyzer.detect_repetitive_patterns()

# Filter patterns by frequency
high_frequency_patterns = [
    p for p in patterns if p['frequency'] >= 5
]
```

### Automation Safety

#### Built-in Safety Features

1. **Failsafe**: Move mouse to screen corner to stop execution
2. **Confirmation**: User confirmation before executing workflows
3. **Coordinate Validation**: Checks if click coordinates are within screen bounds
4. **Application Verification**: Ensures target application is running

#### Customizing Safety Settings

```python
replayer = WorkflowReplayer()

# Disable confirmation prompts (use with caution)
replayer.confirmation_required = False

# Adjust automation speed
replayer.automation_speed = 0.5  # Slower execution

# Disable safety checks (not recommended)
replayer.safety_checks = False
```

### Workflow Optimization

#### Understanding Optimization

The system automatically identifies and removes:
- **Noise Events**: Accidental clicks, random key presses
- **Redundant Actions**: Consecutive similar actions
- **Non-essential Screenshots**: Reduces automation time

#### Manual Optimization

```python
# Create custom optimization rules
def custom_event_filter(event):
    # Custom logic to determine if event is meaningful
    if event['event_type'] == 'mouse':
        # Only include clicks, not movements
        return event.get('mouse_event_type') == 'click'
    return True

# Apply custom filter
optimized_events = [e for e in events if custom_event_filter(e)]
```

## Troubleshooting

### Common Issues

#### 1. OCR Not Working
```bash
# Install Tesseract OCR
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### 2. Permission Errors (macOS)
- Go to System Preferences → Security & Privacy → Privacy
- Enable "Accessibility" for Terminal/Python
- Enable "Screen Recording" for Terminal/Python

#### 3. Automation Not Working
```python
# Check if pyautogui is working
import pyautogui
print(pyautogui.position())  # Should print current mouse position

# Test basic automation
pyautogui.click(100, 100)  # Should click at coordinates
```

#### 4. Database Issues
```bash
# Check database file
sqlite3 screen_data/screen_activity.db ".tables"

# View recent events
sqlite3 screen_data/screen_activity.db "SELECT * FROM system_monitoring ORDER BY timestamp DESC LIMIT 10;"
```

### Performance Optimization

#### Reducing Resource Usage

1. **Increase Screenshot Interval**:
   ```python
   monitor.config["screenshot_interval"] = 10.0  # Every 10 seconds
   ```

2. **Disable OCR for Better Performance**:
   ```python
   monitor.config["ocr_enabled"] = False
   ```

3. **Limit Screenshot Storage**:
   ```python
   monitor.config["save_screenshots"] = False
   ```

#### Database Maintenance

```sql
-- Clean old data (older than 30 days)
DELETE FROM system_monitoring 
WHERE timestamp < datetime('now', '-30 days');

-- Vacuum database to reclaim space
VACUUM;
```

## Best Practices

### Recording Workflows

1. **Plan Your Actions**: Think through the task before recording
2. **Minimize Distractions**: Close unnecessary applications
3. **Use Consistent Timing**: Don't rush or pause unnecessarily
4. **Record Complete Tasks**: Capture entire workflows, not fragments
5. **Test Workflows**: Always test recorded workflows before relying on them

### Analysis and Optimization

1. **Regular Analysis**: Run analysis weekly to identify patterns
2. **Review Documentation**: Check generated docs for accuracy
3. **Optimize Frequently Used Workflows**: Focus on high-impact automations
4. **Monitor Performance**: Track time savings from automation

### Automation Safety

1. **Always Use Dry Run First**: Test workflows before execution
2. **Keep Failsafe Enabled**: Don't disable safety features
3. **Monitor Execution**: Watch automated workflows run
4. **Have Backup Plans**: Know how to manually complete tasks
5. **Regular Backups**: Backup your workflow database

## Integration Examples

### Scheduling Automated Workflows

```bash
# Using cron (macOS/Linux)
# Run workflow every day at 9 AM
0 9 * * * cd /path/to/workflows && python -c "from workflow_replayer import WorkflowReplayer; WorkflowReplayer().replay_workflow('Email_0')"
```

### Custom Triggers

```python
# File watcher trigger
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class WorkflowTrigger(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.csv'):
            # Trigger data processing workflow
            replayer = WorkflowReplayer()
            replayer.replay_workflow('DataProcessing_0')

# Monitor downloads folder
observer = Observer()
observer.schedule(WorkflowTrigger(), '/Users/username/Downloads', recursive=False)
observer.start()
```

## API Reference

For detailed API documentation, see the docstrings in each module:
- `main.py` - ScreenMonitor class
- `workflow_analyzer.py` - WorkflowAnalyzer class  
- `workflow_replayer.py` - WorkflowReplayer class

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files in `analysis_output/` directory
3. Enable debug logging for more detailed information

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```