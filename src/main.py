"""
Screen Monitor - Basic Implementation
A comprehensive system for capturing, processing, and understanding screen activity
"""

import time
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import rumps  # For macOS system tray

# Core libraries for screen capture and processing
try:
    from PIL import ImageGrab, Image
    import cv2
    import numpy as np
    import pytesseract
    from pynput import mouse, keyboard
    import pyaudio
    import wave
    import psutil
except ImportError as e:
    print(f"Missing required library: {e}")
    print(
        "Install with: pip install pillow opencv-python pytesseract pynput pyaudio psutil"
    )
    exit(1)


class ScreenMonitorApp(rumps.App):
    """System tray application for controlling the screen monitor"""

    def __init__(self, monitor):
        super(ScreenMonitorApp, self).__init__("📹")
        self.monitor = monitor
        self.menu = ["Stop Recording"]

    @rumps.clicked("Stop Recording")
    def stop_recording(self, _):
        self.monitor.stop_monitoring()
        rumps.quit_application()


class ScreenMonitor:
    """Main class for monitoring and processing screen activity"""

    def __init__(self, output_dir: str = "screen_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.setup_logging()
        self.setup_database()

        # Control flags
        self.running = False
        self.capture_audio = False

        # Data storage
        self.events = []
        self.screenshots = []

        # Configuration
        self.config = {
            "screenshot_interval": 5.0,  # seconds
            "ocr_enabled": True,
            "save_screenshots": True,
            "audio_duration": 10.0,  # seconds per audio chunk
        }

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "screen_monitor.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Initialize SQLite database for storing events"""
        db_path = self.output_dir / "screen_activity.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        # Create tables
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_monitoring (
                uuid TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                device_name TEXT,
                os_info TEXT,
                monitor_info TEXT,
                keyboard_event_type TEXT,
                keyboard_key TEXT,
                keyboard_details TEXT,
                mouse_event_type TEXT,
                mouse_x INTEGER,
                mouse_y INTEGER,
                mouse_button TEXT,
                mouse_details TEXT,
                active_app TEXT,
                window_title TEXT,
                clipboard_content TEXT,
                screen_content TEXT,
                screenshot_path TEXT,
                location TEXT,
                event_type TEXT
            )
        """
        )

        self.conn.commit()

    def capture_screenshot(self) -> Optional[Dict]:
        """Capture and process a screenshot"""
        try:
            # Capture screen
            screenshot = ImageGrab.grab()
            timestamp = datetime.now().isoformat()

            # Get active window information
            active_window = self.get_active_window()

            # Save screenshot if enabled
            screenshot_path = None
            if self.config["save_screenshots"]:
                filename = f"screenshot_{timestamp.replace(':', '-')}.png"
                screenshot_path = self.output_dir / "screenshots"
                screenshot_path.mkdir(exist_ok=True)
                screenshot_path = screenshot_path / filename
                screenshot.save(screenshot_path)

            # Process with OCR if enabled
            text_content = ""
            if self.config["ocr_enabled"]:
                text_content = self.extract_text_from_image(screenshot)

            # Store in database
            event_data = {
                "timestamp": timestamp,
                "event_type": "screenshot",
                "filename": str(screenshot_path) if screenshot_path else "",
                "text_content": text_content,
                "window_title": active_window.get("title", ""),
                "active_app": active_window.get("app", ""),
            }
            self.store_event(event_data)

            return {
                "timestamp": timestamp,
                "path": str(screenshot_path) if screenshot_path else None,
                "text": text_content,
                "window": active_window,
            }

        except Exception as e:
            self.logger.error(f"Error capturing screenshot: {e}")
            return None

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Preprocess image for better OCR
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply some image processing to improve OCR
            # Increase contrast
            alpha = 1.5  # Contrast control
            beta = 0  # Brightness control
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

            # Use Tesseract to extract text
            text = pytesseract.image_to_string(enhanced)
            return text.strip()

        except Exception as e:
            self.logger.error(f"Error in OCR processing: {e}")
            return ""

    def get_active_window(self) -> Dict:
        """Get information about the currently active window"""
        try:
            # This is a simplified version - you might need platform-specific code
            import subprocess

            # For Windows
            try:
                import win32gui

                hwnd = win32gui.GetForegroundWindow()
                window_title = win32gui.GetWindowText(hwnd)
                return {"title": window_title, "app": "Unknown"}
            except ImportError:
                pass

            # For macOS
            try:
                script = """
                tell application "System Events"
                    set activeApp to name of first application process whose frontmost is true
                    set activeWindow to name of front window of first application process whose frontmost is true
                end tell
                return activeApp & "|" & activeWindow
                """
                result = subprocess.run(
                    ["osascript", "-e", script], capture_output=True, text=True
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split("|")
                    return {
                        "app": parts[0],
                        "title": parts[1] if len(parts) > 1 else "",
                    }
            except:
                pass

            # For Linux
            try:
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return {"title": result.stdout.strip(), "app": "Unknown"}
            except:
                pass

            return {"title": "Unknown", "app": "Unknown"}

        except Exception as e:
            self.logger.error(f"Error getting active window: {e}")
            return {"title": "Error", "app": "Error"}

    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if pressed:
            timestamp = datetime.now().isoformat()
            event_data = {
                "timestamp": timestamp,
                "event_type": "mouse",
                "mouse_event_type": "click",
                "x": x,
                "y": y,
                "button": str(button),
            }
            self.store_event(event_data)
            self.logger.info(f"Mouse click at ({x}, {y}) with {button}")

    def on_mouse_move(self, x, y):
        """Handle mouse move events (optional - can be very verbose)"""
        # Uncomment if you want to track mouse movements
        # timestamp = datetime.now().isoformat()
        # self.store_mouse_event(timestamp, 'move', x, y, '')
        pass

    def on_key_press(self, key):
        """Handle keyboard press events"""
        timestamp = datetime.now().isoformat()
        try:
            key_str = key.char if hasattr(key, "char") else str(key)
        except AttributeError:
            key_str = str(key)

        event_data = {
            "timestamp": timestamp,
            "event_type": "keyboard",
            "keyboard_event_type": "press",
            "key": key_str,
            "text": key_str,
        }
        self.store_event(event_data)
        self.logger.info(f"Key pressed: {key_str}")

    def capture_audio_chunk(self):
        """Capture a chunk of audio"""
        try:
            # Audio recording parameters
            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            rate = 44100
            record_seconds = self.config["audio_duration"]

            p = pyaudio.PyAudio()

            stream = p.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )

            frames = []
            for i in range(0, int(rate / chunk * record_seconds)):
                if not self.running:
                    break
                data = stream.read(chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save audio file
            timestamp = datetime.now().isoformat()
            filename = f"audio_{timestamp.replace(':', '-')}.wav"
            audio_dir = self.output_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            audio_path = audio_dir / filename

            wf = wave.open(str(audio_path), "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
            wf.close()

            self.logger.info(f"Audio saved: {audio_path}")

        except Exception as e:
            self.logger.error(f"Error capturing audio: {e}")

    def store_event(self, event_data):
        """Store event data in unified database table"""
        cursor = self.conn.cursor()

        # Get system information
        import platform
        import socket

        device_name = socket.gethostname()
        os_info = f"{platform.system()} {platform.release()}"

        # Get monitor information
        try:
            from screeninfo import get_monitors

            monitors = get_monitors()
            monitor_info = "; ".join(
                [
                    f"Monitor {i+1}: {m.width}x{m.height} at ({m.x},{m.y})"
                    for i, m in enumerate(monitors)
                ]
            )
        except:
            monitor_info = "Unknown"

        # Generate UUID
        import uuid

        event_uuid = str(uuid.uuid4())

        # Prepare base event data
        base_data = {
            "uuid": event_uuid,
            "timestamp": event_data.get("timestamp"),
            "device_name": device_name,
            "os_info": os_info,
            "monitor_info": monitor_info,
            "event_type": event_data.get("event_type"),
            "active_app": event_data.get("active_app", ""),
            "window_title": event_data.get("window_title", ""),
            "location": "",  # Can be implemented with geolocation if needed
        }

        # Add event-specific data
        if event_data.get("event_type") == "screenshot":
            base_data.update(
                {
                    "screenshot_path": event_data.get("filename", ""),
                    "screen_content": event_data.get("text_content", ""),
                    "clipboard_content": self.get_clipboard_content(),
                }
            )
        elif event_data.get("event_type") == "mouse":
            base_data.update(
                {
                    "mouse_event_type": event_data.get("mouse_event_type", ""),
                    "mouse_x": event_data.get("x"),
                    "mouse_y": event_data.get("y"),
                    "mouse_button": event_data.get("button", ""),
                    "mouse_details": event_data.get("details", ""),
                }
            )
        elif event_data.get("event_type") == "keyboard":
            base_data.update(
                {
                    "keyboard_event_type": event_data.get("keyboard_event_type", ""),
                    "keyboard_key": event_data.get("key", ""),
                    "keyboard_details": event_data.get("text", ""),
                }
            )

        # Create placeholders and values for SQL query
        placeholders = ", ".join(["?" for _ in base_data])
        columns = ", ".join(base_data.keys())
        values = tuple(base_data.values())

        # Insert into database
        cursor.execute(
            f"INSERT INTO system_monitoring ({columns}) VALUES ({placeholders})", values
        )
        self.conn.commit()

    def get_clipboard_content(self):
        """Get current clipboard content"""
        try:
            import pyperclip

            return pyperclip.paste()
        except:
            return ""

    def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            self.logger.warning("Monitoring already running")
            return

        self.running = True
        self.logger.info("Starting screen monitoring...")

        # Setup event listeners
        mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click, on_move=self.on_mouse_move
        )

        keyboard_listener = keyboard.Listener(on_press=self.on_key_press)

        # Start listeners in separate threads
        mouse_listener.start()
        keyboard_listener.start()

        # Screenshot capture thread
        def screenshot_loop():
            while self.running:
                self.capture_screenshot()
                time.sleep(self.config["screenshot_interval"])

        screenshot_thread = threading.Thread(target=screenshot_loop)
        screenshot_thread.daemon = True
        screenshot_thread.start()

        # Audio capture thread (if enabled)
        if self.capture_audio:

            def audio_loop():
                while self.running:
                    self.capture_audio_chunk()

            audio_thread = threading.Thread(target=audio_loop)
            audio_thread.daemon = True
            audio_thread.start()

        self.logger.info("All monitoring threads started")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.logger.info("Stopping screen monitoring...")
        self.running = False
        self.conn.close()
        self.logger.info("Monitoring stopped")

    def generate_report(self):
        """Generate a summary report of captured data"""
        cursor = self.conn.cursor()

        # Get statistics by event type
        cursor.execute(
            "SELECT event_type, COUNT(*) FROM system_monitoring GROUP BY event_type"
        )
        event_counts = dict(cursor.fetchall())

        # Get recent activity across all event types
        cursor.execute(
            """
            SELECT timestamp, event_type, window_title, active_app, monitor_info,
                   keyboard_key, mouse_event_type, mouse_x, mouse_y
            FROM system_monitoring
            ORDER BY timestamp DESC
            LIMIT 10
        """
        )
        recent_events = cursor.fetchall()

        # Get system information
        cursor.execute(
            """
            SELECT device_name, os_info, monitor_info
            FROM system_monitoring
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        system_info = cursor.fetchone()

        report = {
            "summary": {
                "screenshots": event_counts.get("screenshot", 0),
                "mouse_events": event_counts.get("mouse", 0),
                "keyboard_events": event_counts.get("keyboard", 0),
                "generated_at": datetime.now().isoformat(),
                "system_info": {
                    "device_name": system_info[0] if system_info else "Unknown",
                    "os_info": system_info[1] if system_info else "Unknown",
                    "monitor_setup": system_info[2] if system_info else "Unknown",
                },
            },
            "recent_activity": [
                {
                    "timestamp": row[0],
                    "event_type": row[1],
                    "window_title": row[2],
                    "active_app": row[3],
                    "monitor_info": row[4],
                    "details": self._get_event_details(row),
                }
                for row in recent_events
            ],
        }

    def _get_event_details(self, event_row):
        """Helper method to format event details based on event type"""
        event_type = event_row[1]
        if event_type == "keyboard":
            return f"Key: {event_row[5]}"
        elif event_type == "mouse":
            return f"{event_row[6]} at ({event_row[7]}, {event_row[8]})"
        return ""

        # Save report
        report_path = self.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report generated: {report_path}")
        return report


def main():
    """Main function to run the screen monitor"""
    print("Screen Monitor - Basic Implementation")
    print("=====================================")

    # Create monitor instance
    monitor = ScreenMonitor()

    # Configure settings
    monitor.config["screenshot_interval"] = 5.0  # Take screenshot every 10 seconds
    monitor.config["ocr_enabled"] = True  # Enable OCR processing
    monitor.config["save_screenshots"] = True  # Save screenshot files
    monitor.capture_audio = False  # Disable audio capture for now

    print(f"Screenshot interval: {monitor.config['screenshot_interval']} seconds")
    print(f"OCR enabled: {monitor.config['ocr_enabled']}")
    print(f"Audio capture: {monitor.capture_audio}")
    print(
        "\nScreen recording started. Use the system tray icon (📹) to stop recording."
    )

    # Start monitoring in a separate thread
    monitoring_thread = threading.Thread(target=monitor.start_monitoring)
    monitoring_thread.daemon = True
    monitoring_thread.start()

    # Start system tray application
    ScreenMonitorApp(monitor).run()

    # Generate final report
    print("\nGenerating report...")
    report = monitor.generate_report()
    print(f"Captured {report['summary']['screenshots']} screenshots")
    print(f"Recorded {report['summary']['mouse_events']} mouse events")
    print(f"Recorded {report['summary']['keyboard_events']} keyboard events")


if __name__ == "__main__":
    main()
