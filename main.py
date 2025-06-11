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
            CREATE TABLE IF NOT EXISTS screenshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                text_content TEXT,
                window_title TEXT,
                active_app TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mouse_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                x INTEGER,
                y INTEGER,
                button TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS keyboard_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                key TEXT,
                text TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                data TEXT
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
            self.store_screenshot_data(
                timestamp,
                str(screenshot_path) if screenshot_path else "",
                text_content,
                active_window.get("title", ""),
                active_window.get("app", ""),
            )

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
                "event_type": "click",
                "x": x,
                "y": y,
                "button": str(button),
            }

            self.store_mouse_event(timestamp, "click", x, y, str(button))
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

        self.store_keyboard_event(timestamp, "press", key_str, key_str)
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

    def store_screenshot_data(
        self, timestamp, filename, text_content, window_title, active_app
    ):
        """Store screenshot data in database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO screenshots (timestamp, filename, text_content, window_title, active_app)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, filename, text_content, window_title, active_app),
        )
        self.conn.commit()

    def store_mouse_event(self, timestamp, event_type, x, y, button):
        """Store mouse event in database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO mouse_events (timestamp, event_type, x, y, button)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, event_type, x, y, button),
        )
        self.conn.commit()

    def store_keyboard_event(self, timestamp, event_type, key, text):
        """Store keyboard event in database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO keyboard_events (timestamp, event_type, key, text)
            VALUES (?, ?, ?, ?)
        """,
            (timestamp, event_type, key, text),
        )
        self.conn.commit()

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

        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM screenshots")
        screenshot_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM mouse_events")
        mouse_event_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM keyboard_events")
        keyboard_event_count = cursor.fetchone()[0]

        # Get recent activity
        cursor.execute(
            """
            SELECT timestamp, window_title, active_app 
            FROM screenshots 
            ORDER BY timestamp DESC 
            LIMIT 10
        """
        )
        recent_screenshots = cursor.fetchall()

        report = {
            "summary": {
                "screenshots": screenshot_count,
                "mouse_events": mouse_event_count,
                "keyboard_events": keyboard_event_count,
                "generated_at": datetime.now().isoformat(),
            },
            "recent_activity": [
                {"timestamp": row[0], "window_title": row[1], "active_app": row[2]}
                for row in recent_screenshots
            ],
        }

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
    monitor.config["screenshot_interval"] = 10.0  # Take screenshot every 10 seconds
    monitor.config["ocr_enabled"] = True  # Enable OCR processing
    monitor.config["save_screenshots"] = True  # Save screenshot files
    monitor.capture_audio = False  # Disable audio capture for now

    print(f"Output directory: {monitor.output_dir}")
    print(f"Screenshot interval: {monitor.config['screenshot_interval']} seconds")
    print(f"OCR enabled: {monitor.config['ocr_enabled']}")
    print(f"Audio capture: {monitor.capture_audio}")
    print("\nPress Ctrl+C to stop monitoring...")

    try:
        # Start monitoring
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.stop_monitoring()

        # Generate final report
        print("Generating report...")
        report = monitor.generate_report()
        print(f"Captured {report['summary']['screenshots']} screenshots")
        print(f"Recorded {report['summary']['mouse_events']} mouse events")
        print(f"Recorded {report['summary']['keyboard_events']} keyboard events")


if __name__ == "__main__":
    main()
