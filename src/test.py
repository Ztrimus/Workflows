"""
Screen Monitor - Simple Test Version
A minimal implementation to test core functionality
"""

import time
import json
from datetime import datetime
from pathlib import Path
import platform

# Basic libraries - install with: pip install pillow pynput pyautogui
try:
    from PIL import ImageGrab, Image
    from pynput import mouse, keyboard
    import pyautogui

    # Disable pyautogui fail-safe
    pyautogui.FAILSAFE = False
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install with: pip install pillow pynput pyautogui")
    exit()


class SimpleScreenMonitor:
    """Simplified version for testing basic functionality"""

    def __init__(self):
        self.output_dir = Path("simple_screen_data")
        self.output_dir.mkdir(exist_ok=True)

        self.running = False
        self.events = []

        print(f"Output directory: {self.output_dir.absolute()}")

    def get_monitor_info(self):
        """Get information about all connected monitors"""
        monitors = []

        try:
            # Try to get monitor info using different methods
            if platform.system() == "Windows":
                try:
                    import win32api
                    import win32con

                    # Get all monitor handles
                    def enum_display_monitors():
                        monitors = []

                        def callback(hmonitor, hdc, rect, data):
                            monitors.append(
                                {
                                    "left": rect[0],
                                    "top": rect[1],
                                    "right": rect[2],
                                    "bottom": rect[3],
                                    "width": rect[2] - rect[0],
                                    "height": rect[3] - rect[1],
                                }
                            )
                            return True

                        win32api.EnumDisplayMonitors(None, None, callback, 0)
                        return monitors

                    monitors = enum_display_monitors()
                except ImportError:
                    # Fallback method using pyautogui
                    pass

            # Fallback: Use pyautogui to get screen size
            if not monitors:
                # Get primary screen
                primary_width, primary_height = pyautogui.size()
                monitors.append(
                    {
                        "left": 0,
                        "top": 0,
                        "right": primary_width,
                        "bottom": primary_height,
                        "width": primary_width,
                        "height": primary_height,
                        "primary": True,
                    }
                )

                # Try to detect additional monitors
                try:
                    # This is a simple detection - may not work for all setups
                    all_screenshot = ImageGrab.grab(all_screens=True)
                    if all_screenshot.size[0] > primary_width:
                        # Likely multiple monitors
                        total_width = all_screenshot.size[0]
                        monitors = [
                            {
                                "left": 0,
                                "top": 0,
                                "right": total_width,
                                "bottom": primary_height,
                                "width": total_width,
                                "height": primary_height,
                                "primary": True,
                                "multi_monitor": True,
                            }
                        ]
                except:
                    pass

        except Exception as e:
            print(f"Error getting monitor info: {e}")
            # Ultimate fallback
            width, height = pyautogui.size()
            monitors = [
                {
                    "left": 0,
                    "top": 0,
                    "right": width,
                    "bottom": height,
                    "width": width,
                    "height": height,
                    "primary": True,
                }
            ]

        return monitors

    def capture_screenshot(self, monitor_index=None):
        """Take screenshot of specified monitor or all monitors"""
        screenshots_taken = []

        try:
            monitors = self.get_monitor_info()
            print(f"Detected {len(monitors)} monitor(s)")

            if monitor_index is not None and monitor_index < len(monitors):
                # Capture specific monitor
                monitors_to_capture = [monitors[monitor_index]]
                suffix = f"_monitor{monitor_index}"
            else:
                # Capture all monitors
                monitors_to_capture = monitors
                suffix = ""

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, monitor in enumerate(monitors_to_capture):
                try:
                    # Method 1: Try pyautogui (usually more reliable)
                    if len(monitors_to_capture) == 1 and monitor.get(
                        "multi_monitor", False
                    ):
                        # Full multi-monitor screenshot
                        screenshot = pyautogui.screenshot()
                        method = "pyautogui_full"
                    elif len(monitors_to_capture) == 1:
                        # Single monitor with pyautogui
                        screenshot = pyautogui.screenshot(
                            region=(
                                monitor["left"],
                                monitor["top"],
                                monitor["width"],
                                monitor["height"],
                            )
                        )
                        method = "pyautugui_region"
                    else:
                        # Try PIL ImageGrab for specific regions
                        bbox = (
                            monitor["left"],
                            monitor["top"],
                            monitor["right"],
                            monitor["bottom"],
                        )
                        screenshot = ImageGrab.grab(bbox=bbox)
                        method = "imagegrab_bbox"

                    if screenshot:
                        if len(monitors_to_capture) > 1:
                            filename = f"screenshot_{timestamp}_monitor{i}{suffix}.png"
                        else:
                            filename = f"screenshot_{timestamp}{suffix}.png"

                        filepath = self.output_dir / filename
                        screenshot.save(filepath)

                        screenshots_taken.append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "filename": filename,
                                "monitor": i,
                                "size": screenshot.size,
                                "method": method,
                                "monitor_info": monitor,
                            }
                        )

                        print(
                            f"Screenshot saved: {filename} ({screenshot.size[0]}x{screenshot.size[1]}) using {method}"
                        )

                except Exception as e:
                    print(f"Error capturing monitor {i}: {e}")

                    # Fallback: Try alternative method
                    try:
                        if method != "pyautogui_full":
                            screenshot = pyautogui.screenshot()
                            filename = f"screenshot_{timestamp}_fallback.png"
                            filepath = self.output_dir / filename
                            screenshot.save(filepath)

                            screenshots_taken.append(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "filename": filename,
                                    "monitor": "fallback",
                                    "size": screenshot.size,
                                    "method": "pyautogui_fallback",
                                }
                            )
                            print(f"Fallback screenshot saved: {filename}")
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")

            return screenshots_taken if screenshots_taken else None

        except Exception as e:
            print(f"Error in capture_screenshot: {e}")
            return None

    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse clicks"""
        if pressed:  # Only record when button is pressed, not released
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": "mouse_click",
                "x": x,
                "y": y,
                "button": str(button),
            }
            self.events.append(event)
            print(f"Mouse click at ({x}, {y}) with {button}")

    def on_key_press(self, key):
        """Handle key presses"""
        try:
            key_str = key.char if hasattr(key, "char") and key.char else str(key)
        except AttributeError:
            key_str = str(key)

        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "key_press",
            "key": key_str,
        }
        self.events.append(event)
        print(f"Key pressed: {key_str}")

        # Stop monitoring if ESC is pressed
        if key == keyboard.Key.esc:
            print("ESC pressed - stopping monitor...")
            self.running = False
            return False

    def save_events(self):
        """Save all events to JSON file"""
        if self.events:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events_{timestamp}.json"
            filepath = self.output_dir / filename

            with open(filepath, "w") as f:
                json.dump(self.events, f, indent=2)

            print(f"Events saved: {filename} ({len(self.events)} events)")

    def start_simple_test(self, duration=30):
        """Run a simple test for specified duration (seconds)"""
        print(f"\nStarting {duration}-second test...")
        print("- Taking screenshots every 5 seconds")
        print("- Monitoring mouse clicks and key presses")
        print("- Press ESC to stop early")
        print("=" * 50)

        self.running = True

        # Set up input listeners
        mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        keyboard_listener = keyboard.Listener(on_press=self.on_key_press)

        mouse_listener.start()
        keyboard_listener.start()

        start_time = time.time()
        last_screenshot = 0
        screenshot_interval = 5  # seconds

        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()

                # Take screenshot every 5 seconds
                if current_time - last_screenshot >= screenshot_interval:
                    self.capture_screenshot()
                    last_screenshot = current_time

                time.sleep(0.1)  # Small delay to prevent high CPU usage

        except KeyboardInterrupt:
            print("\nKeyboard interrupt - stopping...")

        finally:
            self.running = False
            mouse_listener.stop()
            keyboard_listener.stop()
            self.save_events()

            print("\nTest completed!")
            print(f"Check the '{self.output_dir}' folder for results")


def test_screenshot_only():
    """Test screenshot functionality with multiple options"""
    print("Testing screenshot capture...")
    print("=" * 40)

    # Test 1: Basic pyautogui screenshot
    print("\n1. Testing pyautogui screenshot...")
    try:
        screenshot = pyautogui.screenshot()
        print(
            f"✓ PyAutoGUI screenshot: {screenshot.size[0]}x{screenshot.size[1]} pixels"
        )

        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)

        filename1 = f"test_pyautogui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath1 = output_dir / filename1
        screenshot.save(filepath1)
        print(f"✓ Saved: {filepath1}")

    except Exception as e:
        print(f"✗ PyAutoGUI test failed: {e}")

    # Test 2: PIL ImageGrab
    print("\n2. Testing PIL ImageGrab...")
    try:
        screenshot = ImageGrab.grab()
        print(
            f"✓ PIL ImageGrab screenshot: {screenshot.size[0]}x{screenshot.size[1]} pixels"
        )

        filename2 = f"test_imagegrab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath2 = output_dir / filename2
        screenshot.save(filepath2)
        print(f"✓ Saved: {filepath2}")

    except Exception as e:
        print(f"✗ PIL ImageGrab test failed: {e}")

    # Test 3: All screens with PIL
    print("\n3. Testing all screens capture...")
    try:
        screenshot = ImageGrab.grab(all_screens=True)
        print(
            f"✓ All screens screenshot: {screenshot.size[0]}x{screenshot.size[1]} pixels"
        )

        filename3 = f"test_allscreens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath3 = output_dir / filename3
        screenshot.save(filepath3)
        print(f"✓ Saved: {filepath3}")

    except Exception as e:
        print(f"✗ All screens test failed: {e}")

    # Test 4: Monitor detection and individual capture
    print("\n4. Testing monitor detection...")
    try:
        monitor = SimpleScreenMonitor()
        monitors = monitor.get_monitor_info()
        print(f"✓ Detected {len(monitors)} monitor(s):")

        for i, mon in enumerate(monitors):
            print(
                f"  Monitor {i}: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})"
            )

        # Test individual monitor capture
        print("\n5. Testing individual monitor capture...")
        results = monitor.capture_screenshot()
        if results:
            print(f"✓ Successfully captured {len(results)} screenshot(s)")
            for result in results:
                print(
                    f"  - {result['filename']}: {result['size']} using {result['method']}"
                )
        else:
            print("✗ No screenshots captured")

    except Exception as e:
        print(f"✗ Monitor detection test failed: {e}")

    print(f"\nTest completed! Check the 'test_output' and 'simple_screen_data' folders")
    return True


def test_monitor_specific():
    """Test capturing specific monitors"""
    print("Testing monitor-specific capture...")

    try:
        monitor = SimpleScreenMonitor()
        monitors = monitor.get_monitor_info()

        if len(monitors) > 1:
            print(f"Found {len(monitors)} monitors. Testing individual capture...")

            for i in range(len(monitors)):
                print(f"\nCapturing monitor {i}...")
                result = monitor.capture_screenshot(monitor_index=i)
                if result:
                    print(f"✓ Monitor {i} captured successfully")
                else:
                    print(f"✗ Failed to capture monitor {i}")
        else:
            print("Only one monitor detected. Capturing full screen...")
            result = monitor.capture_screenshot()
            if result:
                print("✓ Screen captured successfully")
            else:
                print("✗ Failed to capture screen")

    except Exception as e:
        print(f"Monitor-specific test failed: {e}")
        return False

    return True


def test_input_monitoring():
    """Test input monitoring for 10 seconds"""
    print("Testing input monitoring for 10 seconds...")
    print("Move mouse and press keys to test")

    events = []

    def on_click(x, y, button, pressed):
        if pressed:
            events.append(f"Click at ({x}, {y}) with {button}")
            print(f"Detected: Click at ({x}, {y}) with {button}")

    def on_key(key):
        events.append(f"Key: {key}")
        print(f"Detected: Key {key}")

    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_key)

    mouse_listener.start()
    keyboard_listener.start()

    time.sleep(10)

    mouse_listener.stop()
    keyboard_listener.stop()

    print(f"\nTest completed! Detected {len(events)} events")
    return len(events) > 0


def main():
    """Main function with different test options"""
    print("Simple Screen Monitor - Test Version")
    print("===================================")

    while True:
        print("\nChoose a test option:")
        print("1. Test screenshot capture (comprehensive)")
        print("2. Test monitor-specific capture")
        print("3. Test input monitoring only")
        print("4. Run full simple monitor (30 seconds)")
        print("5. Run full simple monitor (custom duration)")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            test_screenshot_only()

        elif choice == "2":
            test_monitor_specific()

        elif choice == "3":
            test_input_monitoring()

        elif choice == "4":
            monitor = SimpleScreenMonitor()
            monitor.start_simple_test(30)

        elif choice == "5":
            try:
                duration = int(input("Enter duration in seconds: "))
                monitor = SimpleScreenMonitor()
                monitor.start_simple_test(duration)
            except ValueError:
                print("Please enter a valid number")

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()
