#!/usr/bin/env python3
"""
Manual annotation script for bee crossing events in video files.
Allows marking bee enter/leave events with keyboard controls and rewind functionality.
"""

import cv2
import argparse
import csv
import time
from collections import deque
from typing import List, Tuple

class VideoAnnotator:
    def __init__(self, video_path: str, output_path: str, playback_speed: float = 1.0):
        self.video_path = video_path
        self.output_path = output_path
        self.playback_speed = playback_speed
        self.cap = None
        self.events = []
        self.undo_stack = deque(maxlen=10)  # Keep last 10 events for undo
        
    def open_video(self):
        """Open video file and get properties."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
    def get_current_timestamp(self) -> float:
        """Get current video timestamp in seconds."""
        if self.cap is None:
            return 0.0
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current_frame / self.fps
    
    def add_event(self, event_type: str):
        """Add an event to the list."""
        timestamp = self.get_current_timestamp()
        event = (timestamp, event_type)
        self.events.append(event)
        self.undo_stack.append(event)
        if event_type == "enter":
            print(f"+🐝 at {timestamp:.2f}s")
        else:
            print(f"-🐝 at {timestamp:.2f}s")
    
    def undo_last_events(self, seconds: int = 5):
        """Remove events from the last N seconds."""
        if not self.events:
            print("No events to undo")
            return
            
        current_time = self.get_current_timestamp()
        cutoff_time = current_time - seconds
        
        # Remove events from the last N seconds
        original_count = len(self.events)
        self.events = [event for event in self.events if event[0] <= cutoff_time]
        removed_count = original_count - len(self.events)
        
        if removed_count > 0:
            print(f"Undid {removed_count} events from the last {seconds} seconds")
        else:
            print("No events found in the last 3 seconds to undo")
    
    def rewind_video(self, seconds: int = 3):
        """Rewind video by N seconds."""
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        frames_to_rewind = int(seconds * self.fps)
        new_frame = max(0, current_frame - frames_to_rewind)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        print(f"Rewound video by {seconds} seconds")
    
    def save_events(self):
        """Save events to CSV file."""
        with open(self.output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'event_type'])
            for timestamp, event_type in self.events:
                writer.writerow([timestamp, event_type])
        print(f"Saved {len(self.events)} events to {self.output_path}")
    
    def display_controls(self):
        """Display control instructions."""
        print("\nBee Tunnel Annotation Tool")
        print("Controls:")
        print("  ↑ (Up Arrow)    - Mark bee entering tunnel")
        print("  ↓ (Down Arrow)  - Mark bee leaving tunnel")
        print("  ← (Left Arrow)  - Rewind 3 seconds and undo events")
        print("  ESC             - Exit and save")
        print("  SPACE           - Pause/Resume")
        print()
    
    def run(self):
        """Main annotation loop."""
        self.open_video()
        self.display_controls()
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
            
            # Display current timestamp, total time, and percentage
            timestamp = self.get_current_timestamp()
            percentage = (timestamp / self.duration) * 100 if self.duration > 0 else 0
            cv2.putText(frame, f"Time: {timestamp:.2f}s / {self.duration:.2f}s ({percentage:.1f}%)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Events: {len(self.events)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if paused:
                cv2.putText(frame, "PAUSED", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Bee Tunnel Annotation', frame)
            
            # Handle keyboard input with proper timing
            delay = int(1000 / (self.fps * self.playback_speed)) if not paused else 1
            key = cv2.waitKey(delay) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == 82:  # Up arrow
                self.add_event("enter")
            elif key == 84:  # Down arrow
                self.add_event("leave")
            elif key == 81:  # Left arrow
                self.rewind_video(3)
                self.undo_last_events(3)
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.save_events()

def main():
    parser = argparse.ArgumentParser(description='Manual annotation of bee crossing events in video')
    parser.add_argument('input_path', help='Path to input video file')
    parser.add_argument('output_path', help='Path to output CSV file')
    parser.add_argument('--speed', type=float, default=1.0, 
                       help='Playback speed multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    try:
        annotator = VideoAnnotator(args.input_path, args.output_path, args.speed)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 