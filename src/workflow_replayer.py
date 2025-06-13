"""Workflow Replayer - Automated Task Execution System
Replays analyzed workflows to automate repetitive tasks and replicate user actions.
"""

import json
import time
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# For automation
try:
    import pyautogui
    import psutil
    from pynput import keyboard, mouse
    from pynput.keyboard import Key, Listener as KeyboardListener
    from pynput.mouse import Button, Listener as MouseListener
except ImportError as e:
    print(f"Missing automation libraries: {e}")
    print("Install with: pip install pyautogui psutil pynput")

# For computer vision and screen matching
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageGrab
except ImportError as e:
    print(f"Missing computer vision libraries: {e}")
    print("Install with: pip install opencv-python pillow")


class WorkflowReplayer:
    """Main class for replaying and automating workflows"""

    def __init__(
        self,
        db_path: str = "screen_data/screen_activity.db",
        analysis_dir: str = "analysis_output",
    ):
        self.db_path = Path(db_path)
        self.analysis_dir = Path(analysis_dir)

        self.setup_logging()
        self.load_workflows()

        # Automation settings
        self.automation_speed = 1.0  # Speed multiplier
        self.safety_checks = True
        self.confirmation_required = True

        # Safety settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.1  # Pause between actions

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.analysis_dir / "workflow_replayer.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_workflows(self):
        """Load analyzed workflows from JSON file"""
        try:
            results_file = (
                self.analysis_dir / "reports" / "workflow_analysis_results.json"
            )
            if results_file.exists():
                with open(results_file, "r") as f:
                    self.analysis_results = json.load(f)
                self.workflows = self.analysis_results.get("workflows", {})
                self.logger.info(
                    f"Loaded workflows for {len(self.workflows)} applications"
                )
            else:
                self.workflows = {}
                self.analysis_results = {}
                self.logger.warning("No workflow analysis results found")
        except Exception as e:
            self.logger.error(f"Error loading workflows: {e}")
            self.workflows = {}
            self.analysis_results = {}

    def list_available_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows for replay"""
        workflow_list = []

        for app_name, segments in self.workflows.items():
            for i, segment in enumerate(segments):
                workflow_info = {
                    "id": f"{app_name}_{i}",
                    "application": app_name,
                    "segment_index": i,
                    "duration_minutes": segment.get("duration_seconds", 0) / 60,
                    "event_count": sum(segment.get("event_counts", {}).values()),
                    "description": self._generate_workflow_description(segment),
                    "keywords": segment.get("screen_content_summary", {}).get(
                        "keywords", []
                    ),
                }
                workflow_list.append(workflow_info)

        return workflow_list

    def _generate_workflow_description(self, segment: Dict) -> str:
        """Generate human-readable description of workflow segment"""
        event_counts = segment.get("event_counts", {})
        keywords = segment.get("screen_content_summary", {}).get("keywords", [])
        window_titles = segment.get("window_titles", [])

        description_parts = []

        if event_counts:
            action_summary = []
            if event_counts.get("mouse", 0) > 0:
                action_summary.append(f"{event_counts['mouse']} mouse actions")
            if event_counts.get("keyboard", 0) > 0:
                action_summary.append(f"{event_counts['keyboard']} keyboard inputs")
            if event_counts.get("screenshot", 0) > 0:
                action_summary.append(f"{event_counts['screenshot']} screen captures")

            if action_summary:
                description_parts.append(f"Actions: {', '.join(action_summary)}")

        if window_titles:
            description_parts.append(f"Windows: {', '.join(window_titles[:2])}")

        if keywords:
            description_parts.append(f"Topics: {', '.join(keywords[:3])}")

        return (
            " | ".join(description_parts) if description_parts else "Workflow segment"
        )

    def replay_workflow(
        self, workflow_id: str, speed_multiplier: float = 1.0, dry_run: bool = False
    ) -> bool:
        """Replay a specific workflow"""
        try:
            # Parse workflow ID
            app_name, segment_index = workflow_id.rsplit("_", 1)
            segment_index = int(segment_index)

            if app_name not in self.workflows:
                self.logger.error(f"Application {app_name} not found in workflows")
                return False

            segments = self.workflows[app_name]
            if segment_index >= len(segments):
                self.logger.error(f"Segment index {segment_index} out of range")
                return False

            segment = segments[segment_index]

            self.logger.info(f"Starting replay of workflow: {workflow_id}")

            if dry_run:
                return self._dry_run_workflow(segment)
            else:
                return self._execute_workflow(app_name, segment, speed_multiplier)

        except Exception as e:
            self.logger.error(f"Error replaying workflow {workflow_id}: {e}")
            return False

    def _dry_run_workflow(self, segment: Dict) -> bool:
        """Perform a dry run of the workflow (simulation only)"""
        self.logger.info("=== DRY RUN MODE ===")
        events = segment.get("events", [])

        for i, event in enumerate(events):
            event_type = event.get("event_type")
            timestamp = event.get("timestamp")

            if event_type == "mouse":
                x, y = event.get("mouse_x", 0), event.get("mouse_y", 0)
                button = event.get("mouse_button", "left")
                self.logger.info(
                    f"Step {i+1}: Would click {button} button at ({x}, {y})"
                )

            elif event_type == "keyboard":
                key = event.get("keyboard_key", "")
                self.logger.info(f"Step {i+1}: Would press key '{key}'")

            elif event_type == "screenshot":
                self.logger.info(f"Step {i+1}: Would take screenshot")

            time.sleep(0.1)  # Small delay for readability

        self.logger.info("=== DRY RUN COMPLETE ===")
        return True

    def _execute_workflow(
        self, app_name: str, segment: Dict, speed_multiplier: float
    ) -> bool:
        """Execute the actual workflow"""
        # Safety confirmation
        if self.confirmation_required:
            response = input(f"\nReady to replay workflow for {app_name}? (y/N): ")
            if response.lower() != "y":
                self.logger.info("Workflow replay cancelled by user")
                return False

        # Launch application if needed
        if not self._is_application_running(app_name):
            self.logger.info(f"Launching {app_name}...")
            if not self._launch_application(app_name):
                self.logger.error(f"Failed to launch {app_name}")
                return False
            time.sleep(3)  # Wait for app to load

        # Execute events
        events = segment.get("events", [])
        total_events = len(events)

        self.logger.info(f"Executing {total_events} events...")

        for i, event in enumerate(events):
            try:
                self.logger.info(f"Executing step {i+1}/{total_events}")

                if not self._execute_event(event, speed_multiplier):
                    self.logger.error(f"Failed to execute event {i+1}")
                    return False

                # Add delay between actions
                delay = (1.0 / speed_multiplier) * 0.5
                time.sleep(delay)

            except KeyboardInterrupt:
                self.logger.info("Workflow execution interrupted by user")
                return False
            except Exception as e:
                self.logger.error(f"Error executing event {i+1}: {e}")
                return False

        self.logger.info("Workflow execution completed successfully")
        return True

    def _execute_event(self, event: Dict, speed_multiplier: float) -> bool:
        """Execute a single event"""
        event_type = event.get("event_type")

        if event_type == "mouse":
            return self._execute_mouse_event(event)
        elif event_type == "keyboard":
            return self._execute_keyboard_event(event)
        elif event_type == "screenshot":
            # Skip screenshot events during replay
            return True
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
            return True

    def _execute_mouse_event(self, event: Dict) -> bool:
        """Execute mouse event"""
        try:
            x = event.get("mouse_x", 0)
            y = event.get("mouse_y", 0)
            button = event.get("mouse_button", "Button.left")
            event_subtype = event.get("mouse_event_type", "click")

            # Validate coordinates
            screen_width, screen_height = pyautogui.size()
            if not (0 <= x <= screen_width and 0 <= y <= screen_height):
                self.logger.warning(
                    f"Mouse coordinates ({x}, {y}) out of screen bounds"
                )
                return False

            if event_subtype == "click":
                # Determine button
                if "right" in button.lower():
                    pyautogui.rightClick(x, y)
                elif "middle" in button.lower():
                    pyautogui.middleClick(x, y)
                else:
                    pyautogui.click(x, y)

                self.logger.info(f"Clicked at ({x}, {y})")

            return True

        except Exception as e:
            self.logger.error(f"Error executing mouse event: {e}")
            return False

    def _execute_keyboard_event(self, event: Dict) -> bool:
        """Execute keyboard event"""
        try:
            key = event.get("keyboard_key", "")

            if not key:
                return True

            # Handle special keys
            key_mapping = {
                "Key.space": "space",
                "Key.enter": "enter",
                "Key.tab": "tab",
                "Key.backspace": "backspace",
                "Key.delete": "delete",
                "Key.esc": "escape",
                "Key.shift": "shift",
                "Key.ctrl": "ctrl",
                "Key.alt": "alt",
                "Key.cmd": "cmd",
                "Key.up": "up",
                "Key.down": "down",
                "Key.left": "left",
                "Key.right": "right",
            }

            # Map special keys
            if key in key_mapping:
                pyautogui.press(key_mapping[key])
            elif key.startswith("Key."):
                # Try to extract key name
                key_name = key.replace("Key.", "")
                pyautogui.press(key_name)
            else:
                # Regular character
                if len(key) == 1:
                    pyautogui.press(key)
                else:
                    # Handle multi-character strings (like function keys)
                    pyautogui.press(key)

            self.logger.info(f"Pressed key: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Error executing keyboard event: {e}")
            return False

    def _is_application_running(self, app_name: str) -> bool:
        """Check if application is currently running"""
        try:
            for proc in psutil.process_iter(["pid", "name"]):
                if app_name.lower() in proc.info["name"].lower():
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking if {app_name} is running: {e}")
            return False

    def _launch_application(self, app_name: str) -> bool:
        """Launch application"""
        try:
            # macOS
            result = subprocess.run(
                ["open", "-a", app_name], capture_output=True, text=True
            )
            if result.returncode == 0:
                return True

            # Try alternative approaches
            # Windows
            try:
                subprocess.run(["start", app_name], shell=True, check=True)
                return True
            except:
                pass

            # Linux
            try:
                subprocess.run([app_name], check=True)
                return True
            except:
                pass

            return False

        except Exception as e:
            self.logger.error(f"Error launching {app_name}: {e}")
            return False

    def create_optimized_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Create an optimized version of a workflow by removing noise"""
        try:
            app_name, segment_index = workflow_id.rsplit("_", 1)
            segment_index = int(segment_index)

            if app_name not in self.workflows:
                return None

            segment = self.workflows[app_name][segment_index]
            events = segment.get("events", [])

            # Filter out noise events
            optimized_events = []

            for event in events:
                if self._is_meaningful_event(event):
                    optimized_events.append(event)

            # Remove redundant consecutive events
            optimized_events = self._remove_redundant_events(optimized_events)

            # Create optimized workflow
            optimized_workflow = {
                "original_workflow_id": workflow_id,
                "optimization_timestamp": datetime.now().isoformat(),
                "original_event_count": len(events),
                "optimized_event_count": len(optimized_events),
                "reduction_percentage": (
                    (1 - len(optimized_events) / len(events)) * 100 if events else 0
                ),
                "events": optimized_events,
                "metadata": {
                    "application": app_name,
                    "original_duration": segment.get("duration_seconds", 0),
                    "estimated_optimized_duration": len(optimized_events)
                    * 0.5,  # Rough estimate
                },
            }

            # Save optimized workflow
            optimized_file = (
                self.analysis_dir / "workflows" / f"{workflow_id}_optimized.json"
            )
            with open(optimized_file, "w") as f:
                json.dump(optimized_workflow, f, indent=2, default=str)

            self.logger.info(f"Created optimized workflow: {optimized_file}")
            return optimized_workflow

        except Exception as e:
            self.logger.error(f"Error creating optimized workflow: {e}")
            return None

    def _is_meaningful_event(self, event: Dict) -> bool:
        """Determine if an event is meaningful for workflow automation"""
        event_type = event.get("event_type")

        if event_type == "mouse":
            # Include all mouse clicks
            return event.get("mouse_event_type") == "click"

        elif event_type == "keyboard":
            key = event.get("keyboard_key", "")

            # Exclude very short key presses that might be noise
            if len(key) == 1 and key.isalpha():
                return True

            # Include special keys
            special_keys = [
                "Key.enter",
                "Key.tab",
                "Key.space",
                "Key.backspace",
                "Key.delete",
                "Key.esc",
            ]
            return key in special_keys

        elif event_type == "screenshot":
            # Exclude screenshots from automation
            return False

        return False

    def _remove_redundant_events(self, events: List[Dict]) -> List[Dict]:
        """Remove redundant consecutive events"""
        if not events:
            return events

        filtered_events = [events[0]]

        for i in range(1, len(events)):
            current_event = events[i]
            previous_event = events[i - 1]

            # Check if events are too similar
            if not self._are_events_redundant(previous_event, current_event):
                filtered_events.append(current_event)

        return filtered_events

    def _are_events_redundant(self, event1: Dict, event2: Dict) -> bool:
        """Check if two events are redundant"""
        if event1.get("event_type") != event2.get("event_type"):
            return False

        event_type = event1.get("event_type")

        if event_type == "mouse":
            # Consider mouse clicks redundant if they're very close in position
            x1, y1 = event1.get("mouse_x", 0), event1.get("mouse_y", 0)
            x2, y2 = event2.get("mouse_x", 0), event2.get("mouse_y", 0)
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            return distance < 5  # Less than 5 pixels apart

        elif event_type == "keyboard":
            # Consider keyboard events redundant if they're the same key
            return event1.get("keyboard_key") == event2.get("keyboard_key")

        return False

    def generate_workflow_report(self, workflow_id: str) -> Dict[str, Any]:
        """Generate detailed report for a specific workflow"""
        try:
            app_name, segment_index = workflow_id.rsplit("_", 1)
            segment_index = int(segment_index)

            segment = self.workflows[app_name][segment_index]
            events = segment.get("events", [])

            # Analyze events
            event_analysis = {
                "total_events": len(events),
                "event_types": {},
                "duration_seconds": segment.get("duration_seconds", 0),
                "complexity_score": self._calculate_complexity_score(events),
                "automation_feasibility": self._assess_automation_feasibility(events),
            }

            # Count event types
            for event in events:
                event_type = event.get("event_type")
                event_analysis["event_types"][event_type] = (
                    event_analysis["event_types"].get(event_type, 0) + 1
                )

            report = {
                "workflow_id": workflow_id,
                "application": app_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "event_analysis": event_analysis,
                "optimization_potential": self._analyze_optimization_potential(events),
                "recommendations": self._generate_recommendations(events),
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating workflow report: {e}")
            return {}

    def _calculate_complexity_score(self, events: List[Dict]) -> float:
        """Calculate complexity score for a workflow (0-1 scale)"""
        if not events:
            return 0.0

        # Factors that increase complexity
        unique_positions = set()
        unique_keys = set()

        for event in events:
            if event.get("event_type") == "mouse":
                x, y = event.get("mouse_x", 0), event.get("mouse_y", 0)
                unique_positions.add((x, y))
            elif event.get("event_type") == "keyboard":
                unique_keys.add(event.get("keyboard_key", ""))

        # Normalize complexity score
        position_complexity = min(
            len(unique_positions) / 20, 1.0
        )  # Max 20 unique positions
        key_complexity = min(len(unique_keys) / 30, 1.0)  # Max 30 unique keys
        length_complexity = min(len(events) / 100, 1.0)  # Max 100 events

        return (position_complexity + key_complexity + length_complexity) / 3

    def _assess_automation_feasibility(self, events: List[Dict]) -> str:
        """Assess how feasible it is to automate this workflow"""
        if not events:
            return "Not feasible - no events"

        meaningful_events = [e for e in events if self._is_meaningful_event(e)]
        meaningful_ratio = len(meaningful_events) / len(events)

        if meaningful_ratio > 0.8:
            return "High - mostly meaningful actions"
        elif meaningful_ratio > 0.5:
            return "Medium - some noise present"
        elif meaningful_ratio > 0.2:
            return "Low - significant noise"
        else:
            return "Very Low - mostly noise"

    def _analyze_optimization_potential(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze potential for workflow optimization"""
        if not events:
            return {}

        redundant_count = 0
        noise_count = 0

        for i, event in enumerate(events):
            if not self._is_meaningful_event(event):
                noise_count += 1
            elif i > 0 and self._are_events_redundant(events[i - 1], event):
                redundant_count += 1

        return {
            "noise_events": noise_count,
            "redundant_events": redundant_count,
            "potential_reduction_percentage": (noise_count + redundant_count)
            / len(events)
            * 100,
            "optimized_event_count": len(events) - noise_count - redundant_count,
        }

    def _generate_recommendations(self, events: List[Dict]) -> List[str]:
        """Generate recommendations for workflow improvement"""
        recommendations = []

        if not events:
            return ["No events to analyze"]

        # Analyze patterns and suggest improvements
        meaningful_events = [e for e in events if self._is_meaningful_event(e)]
        meaningful_ratio = len(meaningful_events) / len(events)

        if meaningful_ratio < 0.5:
            recommendations.append(
                "Consider recording workflow more carefully to reduce noise"
            )

        # Check for repetitive patterns
        mouse_events = [e for e in events if e.get("event_type") == "mouse"]
        if len(mouse_events) > 10:
            recommendations.append(
                "High number of mouse clicks - consider keyboard shortcuts"
            )

        # Check for long sequences
        if len(events) > 50:
            recommendations.append(
                "Long workflow - consider breaking into smaller steps"
            )

        # Check for timing
        duration = sum([1 for e in events])  # Rough estimate
        if duration > 100:
            recommendations.append(
                "Consider optimizing for speed by removing unnecessary steps"
            )

        if not recommendations:
            recommendations.append("Workflow appears well-optimized for automation")

        return recommendations


def main():
    """Main function to run workflow replayer"""
    print("Workflow Replayer - Automated Task Execution")
    print("============================================")

    replayer = WorkflowReplayer()

    # List available workflows
    workflows = replayer.list_available_workflows()

    if not workflows:
        print("No workflows found. Please run workflow analysis first.")
        return

    print(f"\nFound {len(workflows)} available workflows:")
    print("-" * 60)

    for i, workflow in enumerate(workflows):
        print(f"{i+1}. {workflow['id']}")
        print(f"   App: {workflow['application']}")
        print(f"   Duration: {workflow['duration_minutes']:.1f} minutes")
        print(f"   Events: {workflow['event_count']}")
        print(f"   Description: {workflow['description']}")
        print()

    # Interactive workflow selection
    try:
        choice = input("Select workflow to replay (number) or 'q' to quit: ")

        if choice.lower() == "q":
            return

        workflow_index = int(choice) - 1
        if 0 <= workflow_index < len(workflows):
            selected_workflow = workflows[workflow_index]
            workflow_id = selected_workflow["id"]

            print(f"\nSelected: {workflow_id}")

            # Options
            print("\nOptions:")
            print("1. Dry run (simulation only)")
            print("2. Execute workflow")
            print("3. Create optimized version")
            print("4. Generate report")

            option = input("Choose option (1-4): ")

            if option == "1":
                replayer.replay_workflow(workflow_id, dry_run=True)
            elif option == "2":
                speed = float(input("Speed multiplier (1.0 = normal): ") or "1.0")
                replayer.replay_workflow(workflow_id, speed_multiplier=speed)
            elif option == "3":
                optimized = replayer.create_optimized_workflow(workflow_id)
                if optimized:
                    print(f"Optimized workflow created:")
                    print(f"Original events: {optimized['original_event_count']}")
                    print(f"Optimized events: {optimized['optimized_event_count']}")
                    print(f"Reduction: {optimized['reduction_percentage']:.1f}%")
            elif option == "4":
                report = replayer.generate_workflow_report(workflow_id)
                print(json.dumps(report, indent=2, default=str))

        else:
            print("Invalid selection")

    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == "__main__":
    main()
