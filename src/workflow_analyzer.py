"""Workflow Analyzer - Advanced Analysis and Documentation System
Analyzes screen activity data to understand user workflows, create documentation,
and enable workflow automation and replication.
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict, Counter
import re

# For visualization and knowledge graphs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    from wordcloud import WordCloud
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"Missing visualization libraries: {e}")
    print("Install with: pip install matplotlib seaborn networkx wordcloud plotly")

# For NLP and text analysis
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import spacy
except ImportError as e:
    print(f"Missing NLP libraries: {e}")
    print("Install with: pip install scikit-learn spacy")


class WorkflowAnalyzer:
    """Main class for analyzing screen activity and extracting workflows"""

    def __init__(
        self,
        db_path: str = "screen_data/screen_activity.db",
        output_dir: str = "analysis_output",
    ):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for different types of output
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "workflows").mkdir(exist_ok=True)
        (self.output_dir / "documentation").mkdir(exist_ok=True)

        self.setup_logging()
        self.load_data()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "workflow_analyzer.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load data from the screen activity database"""
        try:
            self.conn = sqlite3.connect(self.db_path)

            # Load all data into pandas DataFrame for easier analysis
            query = """
            SELECT * FROM system_monitoring 
            ORDER BY timestamp
            """
            self.df = pd.read_sql_query(query, self.conn)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

            self.logger.info(f"Loaded {len(self.df)} events from database")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def identify_sessions(self, idle_threshold_minutes: int = 30) -> List[Dict]:
        """Identify user sessions based on activity gaps"""
        if self.df.empty:
            return []

        sessions = []
        current_session_start = None
        last_activity = None

        for _, row in self.df.iterrows():
            current_time = row["timestamp"]

            if last_activity is None:
                # First event
                current_session_start = current_time
            else:
                # Check if gap is larger than threshold
                gap = (current_time - last_activity).total_seconds() / 60
                if gap > idle_threshold_minutes:
                    # End current session and start new one
                    if current_session_start:
                        sessions.append(
                            {
                                "start_time": current_session_start,
                                "end_time": last_activity,
                                "duration_minutes": (
                                    last_activity - current_session_start
                                ).total_seconds()
                                / 60,
                            }
                        )
                    current_session_start = current_time

            last_activity = current_time

        # Add final session
        if current_session_start and last_activity:
            sessions.append(
                {
                    "start_time": current_session_start,
                    "end_time": last_activity,
                    "duration_minutes": (
                        last_activity - current_session_start
                    ).total_seconds()
                    / 60,
                }
            )

        self.logger.info(f"Identified {len(sessions)} user sessions")
        return sessions

    def extract_application_workflows(self) -> Dict[str, List[Dict]]:
        """Extract workflows grouped by application"""
        workflows = defaultdict(list)

        if self.df.empty:
            return dict(workflows)

        # Group events by application and time proximity
        app_groups = self.df.groupby("active_app")

        for app_name, app_data in app_groups:
            if pd.isna(app_name) or app_name == "Unknown":
                continue

            # Sort by timestamp
            app_data = app_data.sort_values("timestamp")

            # Identify workflow segments within the app
            workflow_segments = self._segment_app_workflow(app_data)
            workflows[app_name] = workflow_segments

        return dict(workflows)

    def _segment_app_workflow(
        self, app_data: pd.DataFrame, gap_threshold_minutes: int = 10
    ) -> List[Dict]:
        """Segment application data into workflow steps"""
        segments = []
        current_segment = []
        last_time = None

        for _, row in app_data.iterrows():
            current_time = row["timestamp"]

            if (
                last_time is None
                or (current_time - last_time).total_seconds() / 60
                <= gap_threshold_minutes
            ):
                current_segment.append(row.to_dict())
            else:
                # Save current segment and start new one
                if current_segment:
                    segments.append(self._analyze_segment(current_segment))
                current_segment = [row.to_dict()]

            last_time = current_time

        # Add final segment
        if current_segment:
            segments.append(self._analyze_segment(current_segment))

        return segments

    def _analyze_segment(self, segment_events: List[Dict]) -> Dict:
        """Analyze a workflow segment to extract patterns"""
        if not segment_events:
            return {}

        start_time = segment_events[0]["timestamp"]
        end_time = segment_events[-1]["timestamp"]

        # Count event types
        event_counts = Counter([event["event_type"] for event in segment_events])

        # Extract key actions
        key_actions = []
        for event in segment_events:
            if event["event_type"] == "keyboard" and event["keyboard_key"]:
                key_actions.append(event["keyboard_key"])
            elif (
                event["event_type"] == "mouse" and event["mouse_event_type"] == "click"
            ):
                key_actions.append(f"click_at_{event['mouse_x']}_{event['mouse_y']}")

        # Extract window titles and screen content
        window_titles = [
            event["window_title"] for event in segment_events if event["window_title"]
        ]
        screen_contents = [
            event["screen_content"]
            for event in segment_events
            if event["screen_content"]
        ]

        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": (
                pd.to_datetime(end_time) - pd.to_datetime(start_time)
            ).total_seconds(),
            "event_counts": dict(event_counts),
            "key_actions": key_actions,
            "window_titles": list(set(window_titles)),
            "screen_content_summary": self._summarize_screen_content(screen_contents),
            "events": segment_events,
        }

    def _summarize_screen_content(self, screen_contents: List[str]) -> Dict:
        """Summarize screen content using NLP techniques"""
        if not screen_contents:
            return {"keywords": [], "summary": ""}

        # Combine all screen content
        combined_text = " ".join([content for content in screen_contents if content])

        if not combined_text.strip():
            return {"keywords": [], "summary": ""}

        try:
            # Extract keywords using TF-IDF
            vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()

            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keywords = [
                (feature_names[i], scores[i]) for i in scores.argsort()[-10:][::-1]
            ]

            return {
                "keywords": [kw[0] for kw in keywords if kw[1] > 0],
                "summary": (
                    combined_text[:200] + "..."
                    if len(combined_text) > 200
                    else combined_text
                ),
                "word_count": len(combined_text.split()),
            }
        except Exception as e:
            self.logger.error(f"Error in content summarization: {e}")
            return {"keywords": [], "summary": combined_text[:200] + "..."}

    def detect_repetitive_patterns(self) -> List[Dict]:
        """Detect repetitive patterns in user behavior"""
        patterns = []

        if self.df.empty:
            return patterns

        # Group by application and look for repetitive sequences
        app_groups = self.df.groupby("active_app")

        for app_name, app_data in app_groups:
            if pd.isna(app_name) or len(app_data) < 5:
                continue

            # Create sequence of actions
            action_sequence = []
            for _, row in app_data.iterrows():
                if row["event_type"] == "keyboard" and row["keyboard_key"]:
                    action_sequence.append(f"key_{row['keyboard_key']}")
                elif (
                    row["event_type"] == "mouse" and row["mouse_event_type"] == "click"
                ):
                    action_sequence.append(f"click_{row['mouse_x']}_{row['mouse_y']}")

            # Find repetitive subsequences
            repetitive_patterns = self._find_repetitive_subsequences(action_sequence)

            for pattern in repetitive_patterns:
                patterns.append(
                    {
                        "application": app_name,
                        "pattern": pattern["sequence"],
                        "frequency": pattern["count"],
                        "confidence": (
                            pattern["count"] / len(action_sequence)
                            if action_sequence
                            else 0
                        ),
                    }
                )

        return patterns

    def _find_repetitive_subsequences(
        self, sequence: List[str], min_length: int = 3, min_frequency: int = 2
    ) -> List[Dict]:
        """Find repetitive subsequences in a sequence of actions"""
        patterns = []
        sequence_length = len(sequence)

        # Check subsequences of different lengths
        for length in range(min_length, min(sequence_length // 2 + 1, 10)):
            subsequence_counts = Counter()

            for i in range(sequence_length - length + 1):
                subsequence = tuple(sequence[i : i + length])
                subsequence_counts[subsequence] += 1

            # Find patterns that repeat at least min_frequency times
            for subsequence, count in subsequence_counts.items():
                if count >= min_frequency:
                    patterns.append(
                        {
                            "sequence": list(subsequence),
                            "count": count,
                            "length": length,
                        }
                    )

        return patterns

    def generate_workflow_documentation(
        self, workflows: Dict[str, List[Dict]]
    ) -> Dict[str, str]:
        """Generate human-readable documentation for workflows"""
        documentation = {}

        for app_name, segments in workflows.items():
            if not segments:
                continue

            doc_lines = []
            doc_lines.append(f"# Workflow Documentation: {app_name}")
            doc_lines.append(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            doc_lines.append("")

            # Summary statistics
            total_segments = len(segments)
            total_duration = sum([seg.get("duration_seconds", 0) for seg in segments])

            doc_lines.append("## Summary")
            doc_lines.append(f"- Total workflow segments: {total_segments}")
            doc_lines.append(f"- Total time spent: {total_duration / 60:.1f} minutes")
            doc_lines.append(
                f"- Average segment duration: {total_duration / total_segments / 60:.1f} minutes"
            )
            doc_lines.append("")

            # Detailed workflow steps
            doc_lines.append("## Workflow Steps")

            for i, segment in enumerate(segments, 1):
                doc_lines.append(f"### Step {i}")
                doc_lines.append(
                    f"**Duration:** {segment.get('duration_seconds', 0) / 60:.1f} minutes"
                )

                if segment.get("window_titles"):
                    doc_lines.append(
                        f"**Windows:** {', '.join(segment['window_titles'])}"
                    )

                if segment.get("screen_content_summary", {}).get("keywords"):
                    keywords = segment["screen_content_summary"]["keywords"]
                    doc_lines.append(f"**Key topics:** {', '.join(keywords[:5])}")

                # Action summary
                event_counts = segment.get("event_counts", {})
                if event_counts:
                    doc_lines.append(f"**Actions:** {dict(event_counts)}")

                doc_lines.append("")

            documentation[app_name] = "\n".join(doc_lines)

        return documentation

    def create_workflow_visualization(self, workflows: Dict[str, List[Dict]]) -> None:
        """Create visualizations for workflow analysis"""
        try:
            # 1. Application usage timeline
            self._create_timeline_visualization()

            # 2. Workflow network graph
            self._create_workflow_network(workflows)

            # 3. Activity heatmap
            self._create_activity_heatmap()

            # 4. Keyword cloud
            self._create_keyword_cloud(workflows)

            self.logger.info("Visualizations created successfully")

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")

    def _create_timeline_visualization(self):
        """Create timeline visualization of application usage"""
        if self.df.empty:
            return

        # Prepare data for timeline
        timeline_data = (
            self.df.groupby(["active_app", pd.Grouper(key="timestamp", freq="H")])
            .size()
            .reset_index(name="event_count")
        )

        # Create interactive timeline with Plotly
        fig = px.timeline(
            timeline_data,
            x_start="timestamp",
            x_end="timestamp",
            y="active_app",
            color="event_count",
            title="Application Usage Timeline",
        )

        fig.write_html(self.output_dir / "visualizations" / "timeline.html")

    def _create_workflow_network(self, workflows: Dict[str, List[Dict]]):
        """Create network graph showing workflow transitions"""
        G = nx.DiGraph()

        for app_name, segments in workflows.items():
            for i, segment in enumerate(segments):
                node_id = f"{app_name}_step_{i+1}"
                G.add_node(
                    node_id,
                    app=app_name,
                    duration=segment.get("duration_seconds", 0),
                    keywords=segment.get("screen_content_summary", {}).get(
                        "keywords", []
                    ),
                )

                # Add edges between consecutive steps
                if i > 0:
                    prev_node = f"{app_name}_step_{i}"
                    G.add_edge(prev_node, node_id)

        # Save network graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=1000,
            font_size=8,
            arrows=True,
        )
        plt.title("Workflow Network Graph")
        plt.savefig(
            self.output_dir / "visualizations" / "workflow_network.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_activity_heatmap(self):
        """Create heatmap of activity patterns"""
        if self.df.empty:
            return

        # Create hour and day columns
        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["day_of_week"] = self.df["timestamp"].dt.day_name()

        # Create pivot table for heatmap
        heatmap_data = (
            self.df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
        )

        # Reorder days
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heatmap_data = heatmap_data.reindex(day_order)

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Activity Heatmap (Events per Hour)")
        plt.ylabel("Day of Week")
        plt.xlabel("Hour of Day")
        plt.savefig(
            self.output_dir / "visualizations" / "activity_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_keyword_cloud(self, workflows: Dict[str, List[Dict]]):
        """Create word cloud from screen content"""
        all_keywords = []

        for app_name, segments in workflows.items():
            for segment in segments:
                keywords = segment.get("screen_content_summary", {}).get("keywords", [])
                all_keywords.extend(keywords)

        if all_keywords:
            keyword_text = " ".join(all_keywords)
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(keyword_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Workflow Keywords Cloud")
            plt.savefig(
                self.output_dir / "visualizations" / "keyword_cloud.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def generate_automation_scripts(
        self, workflows: Dict[str, List[Dict]]
    ) -> Dict[str, str]:
        """Generate automation scripts for repetitive workflows"""
        automation_scripts = {}

        for app_name, segments in workflows.items():
            if not segments:
                continue

            # Find the most common workflow pattern
            most_common_pattern = self._find_most_common_pattern(segments)

            if most_common_pattern:
                script = self._generate_automation_script(app_name, most_common_pattern)
                automation_scripts[app_name] = script

        return automation_scripts

    def _find_most_common_pattern(self, segments: List[Dict]) -> Optional[Dict]:
        """Find the most common workflow pattern in segments"""
        if not segments:
            return None

        # For now, return the longest segment as the most representative
        return max(segments, key=lambda x: x.get("duration_seconds", 0))

    def _generate_automation_script(self, app_name: str, pattern: Dict) -> str:
        """Generate Python automation script for a workflow pattern"""
        script_lines = []
        script_lines.append(f'"""Automation script for {app_name} workflow"""')
        script_lines.append("import time")
        script_lines.append("import pyautogui")
        script_lines.append("import subprocess")
        script_lines.append("")
        script_lines.append("def automate_workflow():")
        script_lines.append(f'    """Automate {app_name} workflow"""')
        script_lines.append("    ")
        script_lines.append(f"    # Launch {app_name}")
        script_lines.append(
            f'    # subprocess.run(["open", "-a", "{app_name}"])  # macOS'
        )
        script_lines.append("    time.sleep(2)  # Wait for app to load")
        script_lines.append("    ")

        # Add actions based on the pattern
        events = pattern.get("events", [])
        for i, event in enumerate(events[:10]):  # Limit to first 10 events
            if event["event_type"] == "mouse" and event["mouse_event_type"] == "click":
                x, y = event.get("mouse_x", 0), event.get("mouse_y", 0)
                script_lines.append(f"    # Step {i+1}: Click at ({x}, {y})")
                script_lines.append(f"    pyautogui.click({x}, {y})")
                script_lines.append("    time.sleep(0.5)")
            elif event["event_type"] == "keyboard" and event.get("keyboard_key"):
                key = event["keyboard_key"]
                script_lines.append(f"    # Step {i+1}: Press key {key}")
                script_lines.append(f'    pyautogui.press("{key}")')
                script_lines.append("    time.sleep(0.2)")

        script_lines.append("")
        script_lines.append('if __name__ == "__main__":')
        script_lines.append("    automate_workflow()")

        return "\n".join(script_lines)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete workflow analysis pipeline"""
        self.logger.info("Starting complete workflow analysis...")

        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_events": len(self.df),
                "date_range": {
                    "start": (
                        self.df["timestamp"].min().isoformat()
                        if not self.df.empty
                        else None
                    ),
                    "end": (
                        self.df["timestamp"].max().isoformat()
                        if not self.df.empty
                        else None
                    ),
                },
                "applications": (
                    self.df["active_app"].nunique() if not self.df.empty else 0
                ),
            },
        }

        # 1. Identify sessions
        sessions = self.identify_sessions()
        results["sessions"] = sessions

        # 2. Extract workflows
        workflows = self.extract_application_workflows()
        results["workflows"] = workflows

        # 3. Detect patterns
        patterns = self.detect_repetitive_patterns()
        results["repetitive_patterns"] = patterns

        # 4. Generate documentation
        documentation = self.generate_workflow_documentation(workflows)
        results["documentation"] = documentation

        # 5. Create visualizations
        self.create_workflow_visualization(workflows)

        # 6. Generate automation scripts
        automation_scripts = self.generate_automation_scripts(workflows)
        results["automation_scripts"] = automation_scripts

        # Save results
        self._save_analysis_results(results)

        self.logger.info("Complete workflow analysis finished")
        return results

    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to files"""
        # Save main results as JSON
        results_file = self.output_dir / "reports" / "workflow_analysis_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save documentation as markdown files
        for app_name, doc_content in results.get("documentation", {}).items():
            doc_file = (
                self.output_dir
                / "documentation"
                / f"{app_name.replace(' ', '_')}_workflow.md"
            )
            with open(doc_file, "w") as f:
                f.write(doc_content)

        # Save automation scripts
        for app_name, script_content in results.get("automation_scripts", {}).items():
            script_file = (
                self.output_dir
                / "workflows"
                / f"{app_name.replace(' ', '_')}_automation.py"
            )
            with open(script_file, "w") as f:
                f.write(script_content)

        self.logger.info(f"Analysis results saved to {self.output_dir}")


def main():
    """Main function to run workflow analysis"""
    print("Workflow Analyzer - Advanced Analysis System")
    print("============================================")

    # Initialize analyzer
    analyzer = WorkflowAnalyzer()

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    # Print summary
    print(f"\nAnalysis Complete!")
    print(f"Total events analyzed: {results['data_summary']['total_events']}")
    print(f"Applications found: {results['data_summary']['applications']}")
    print(f"Sessions identified: {len(results['sessions'])}")
    print(f"Workflows extracted: {len(results['workflows'])}")
    print(f"Repetitive patterns found: {len(results['repetitive_patterns'])}")

    print(f"\nResults saved to: {analyzer.output_dir}")
    print("- Reports: analysis_output/reports/")
    print("- Documentation: analysis_output/documentation/")
    print("- Visualizations: analysis_output/visualizations/")
    print("- Automation Scripts: analysis_output/workflows/")


if __name__ == "__main__":
    main()
