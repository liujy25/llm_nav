#!/usr/bin/env python3
"""
Server Performance Test Script
Tests the navigation server with synthetic data for 20 iterations
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List
import matplotlib.pyplot as plt


class ServerTester:
    def __init__(self, server_url: str = "http://10.19.126.158:1874"):
        self.server_url = server_url.rstrip('/')
        self.timing_records = []
        
    def generate_fake_rgb(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generate fake RGB image with some patterns"""
        # Create a colorful pattern
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(height):
            for j in range(width):
                rgb[i, j, 0] = int((i / height) * 255)  # Red gradient
                rgb[i, j, 1] = int((j / width) * 255)   # Green gradient
                rgb[i, j, 2] = 128                       # Blue constant
        
        # Add some random rectangles to simulate objects
        num_objects = np.random.randint(3, 8)
        for _ in range(num_objects):
            x1 = np.random.randint(0, width - 50)
            y1 = np.random.randint(0, height - 50)
            x2 = x1 + np.random.randint(30, 100)
            y2 = y1 + np.random.randint(30, 100)
            color = np.random.randint(0, 256, 3)
            rgb[y1:y2, x1:x2] = color
        
        return rgb
    
    def generate_fake_depth(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generate fake depth image"""
        # Create depth with some variation
        depth = np.random.uniform(0.5, 3.0, (height, width)).astype(np.float32)
        
        # Add some closer regions (simulate objects)
        num_objects = np.random.randint(2, 5)
        for _ in range(num_objects):
            x1 = np.random.randint(0, width - 50)
            y1 = np.random.randint(0, height - 50)
            x2 = x1 + np.random.randint(30, 80)
            y2 = y1 + np.random.randint(30, 80)
            depth[y1:y2, x1:x2] = np.random.uniform(0.3, 1.5)
        
        return depth
    
    def generate_fake_intrinsic(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generate fake camera intrinsic matrix"""
        fx = width * 0.8  # typical focal length
        fy = height * 0.8
        cx = width / 2.0
        cy = height / 2.0
        
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return K
    
    def generate_fake_transforms(self, iteration: int):
        """Generate fake transformation matrices with slight variations"""
        # Base pose that changes slightly each iteration
        base_x = 0.0 + iteration * 0.1  # Move forward 10cm each iteration
        base_y = 0.0 + np.random.uniform(-0.05, 0.05)  # Small random lateral movement
        base_yaw = np.random.uniform(-0.1, 0.1)  # Small random rotation
        
        # T_cam_odom: odom -> camera
        # Camera is typically offset from base
        cam_height = 1.2
        cam_forward = 0.1
        
        c = np.cos(base_yaw)
        s = np.sin(base_yaw)
        
        # Camera position in odom frame
        cam_x = base_x + cam_forward * c
        cam_y = base_y + cam_forward * s
        cam_z = cam_height
        
        # Camera looks forward (pitched down slightly)
        pitch = -0.2  # Looking slightly down
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(base_yaw)
        sy = np.sin(base_yaw)
        
        # Rotation matrix for camera (yaw around Z, then pitch around Y)
        R_cam = np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp, cy, sy * sp],
            [-sp, 0, cp]
        ])
        
        T_cam_odom = np.eye(4, dtype=np.float32)
        T_cam_odom[:3, :3] = R_cam
        T_cam_odom[:3, 3] = [cam_x, cam_y, cam_z]
        T_cam_odom = np.linalg.inv(T_cam_odom)  # odom->cam
        
        # T_odom_base: base -> odom
        T_odom_base = np.eye(4, dtype=np.float32)
        T_odom_base[:3, :3] = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        T_odom_base[:3, 3] = [base_x, base_y, 0.0]
        
        return T_cam_odom, T_odom_base
    
    def test_health(self) -> bool:
        """Test if server is healthy"""
        try:
            response = requests.get(f'{self.server_url}/health', timeout=5)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Server health check: {result}")
            return True
        except Exception as e:
            print(f"✗ Server health check failed: {e}")
            return False
    
    def reset_navigation(self, goal: str = "test_object", goal_description: str = ""):
        """Reset navigation on server"""
        url = f'{self.server_url}/navigation_reset'
        data = {
            'goal': goal,
            'goal_description': goal_description,
            'confidence_threshold': 0.5
        }
        
        print(f"\n{'='*60}")
        print(f"Resetting navigation: goal='{goal}'")
        print(f"{'='*60}")
        
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Navigation reset successful: {result.get('log_dir', 'N/A')}")
            return result
        except Exception as e:
            print(f"✗ Navigation reset failed: {e}")
            raise
    
    def send_navigation_step(self, iteration: int) -> Dict[str, Any]:
        """Send a single navigation step with fake data"""
        url = f'{self.server_url}/navigation_step'
        
        # Generate fake data
        rgb = self.generate_fake_rgb()
        depth = self.generate_fake_depth()
        intrinsic = self.generate_fake_intrinsic()
        T_cam_odom, T_odom_base = self.generate_fake_transforms(iteration)
        
        # Prepare RGB image (JPEG)
        rgb_pil = Image.fromarray(rgb)
        rgb_buffer = BytesIO()
        rgb_pil.save(rgb_buffer, format='JPEG', quality=95)
        rgb_buffer.seek(0)
        
        # Prepare Depth image (PNG 16-bit)
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_mm, mode='I;16')
        depth_buffer = BytesIO()
        depth_pil.save(depth_buffer, format='PNG')
        depth_buffer.seek(0)
        
        # Prepare form data
        files = {
            'rgb': ('rgb.jpg', rgb_buffer, 'image/jpeg'),
            'depth': ('depth.png', depth_buffer, 'image/png')
        }
        
        data = {
            'intrinsic': json.dumps(intrinsic.flatten().tolist()),
            'T_cam_odom': json.dumps(T_cam_odom.flatten().tolist()),
            'T_odom_base': json.dumps(T_odom_base.flatten().tolist())
        }
        
        print(f"\n{'─'*60}")
        print(f"Iteration {iteration}")
        print(f"{'─'*60}")
        
        try:
            t_start = time.time()
            response = requests.post(url, files=files, data=data, timeout=120)
            t_end = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            # Calculate timing
            total_time = t_end - t_start
            timing = result.get('timing', {})
            server_time = timing.get('total_server_time', 0.0)
            network_time = total_time - server_time
            
            # Store timing record
            record = {
                'iteration': iteration,
                'total_time': total_time,
                'server_time': server_time,
                'network_time': network_time,
                'detection_time': timing.get('detection_time', 0.0),
                'vlm_projection_time': timing.get('vlm_projection_time', 0.0),
                'vlm_inference_time': timing.get('vlm_inference_time', 0.0),
                'action_type': result.get('action_type', 'none'),
                'detected': result.get('detected', False),
                'detection_score': result.get('detection_score', 0.0)
            }
            self.timing_records.append(record)
            
            # Print results
            print(f"Action: {result.get('action_type', 'none')}")
            print(f"Detected: {result.get('detected', False)}, Score: {result.get('detection_score', 0.0):.3f}")
            print(f"Total: {total_time:.3f}s | Server: {server_time:.3f}s | Network: {network_time:.3f}s")
            print(f"  Detection: {timing.get('detection_time', 0.0):.3f}s | "
                  f"Projection: {timing.get('vlm_projection_time', 0.0):.3f}s | "
                  f"VLM: {timing.get('vlm_inference_time', 0.0):.3f}s")
            
            return result
            
        except Exception as e:
            print(f"✗ Request failed: {e}")
            raise
    
    def print_statistics(self):
        """Print statistics from all timing records"""
        if not self.timing_records:
            print("\nNo timing records to analyze.")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE TEST RESULTS")
        print("="*80)
        
        # Calculate statistics
        total_times = [r['total_time'] for r in self.timing_records]
        server_times = [r['server_time'] for r in self.timing_records]
        network_times = [r['network_time'] for r in self.timing_records]
        detection_times = [r['detection_time'] for r in self.timing_records]
        projection_times = [r['vlm_projection_time'] for r in self.timing_records]
        inference_times = [r['vlm_inference_time'] for r in self.timing_records]
        
        detected_count = sum(1 for r in self.timing_records if r['detected'])
        action_types = {}
        for r in self.timing_records:
            action = r['action_type']
            action_types[action] = action_types.get(action, 0) + 1
        
        print(f"\nTotal Iterations: {len(self.timing_records)}")
        print(f"Target Detected: {detected_count} times ({detected_count/len(self.timing_records)*100:.1f}%)")
        
        print("\n" + "-"*80)
        print("ACTION TYPE DISTRIBUTION:")
        print("-"*80)
        for action_type, count in sorted(action_types.items()):
            percentage = count / len(self.timing_records) * 100
            print(f"  {action_type:20s}: {count:3d} times ({percentage:5.1f}%)")
        
        print("\n" + "-"*80)
        print("TIMING STATISTICS (seconds):")
        print("-"*80)
        
        def print_stats(label, values):
            mean = np.mean(values)
            median = np.median(values)
            min_val = np.min(values)
            max_val = np.max(values)
            std_val = np.std(values)
            total = np.sum(values)
            print(f"{label:<25} {mean:>8.3f} {median:>8.3f} {min_val:>8.3f} "
                  f"{max_val:>8.3f} {std_val:>8.3f} {total:>10.3f}")
        
        print(f"{'Category':<25} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'Std':>8} {'Total':>10}")
        print("-"*80)
        print_stats("Total Time", total_times)
        print_stats("Server Time", server_times)
        print_stats("Network Time", network_times)
        print_stats("Detection Time", detection_times)
        print_stats("VLM Projection Time", projection_times)
        print_stats("VLM Inference Time", inference_times)
        
        # Calculate breakdown percentages
        mean_server = np.mean(server_times)
        mean_network = np.mean(network_times)
        mean_detection = np.mean(detection_times)
        mean_projection = np.mean(projection_times)
        mean_inference = np.mean(inference_times)
        mean_total = np.mean(total_times)
        
        print("\n" + "-"*80)
        print("TIME BREAKDOWN (percentage of total time):")
        print("-"*80)
        print(f"  Server Processing:  {mean_server/mean_total*100:6.2f}%")
        print(f"    └─ Detection:     {mean_detection/mean_total*100:6.2f}%")
        print(f"    └─ VLM Projection:{mean_projection/mean_total*100:6.2f}%")
        print(f"    └─ VLM Inference: {mean_inference/mean_total*100:6.2f}%")
        print(f"  Network Transfer:   {mean_network/mean_total*100:6.2f}%")
        
        print("\n" + "="*80)
    
    def save_results(self, output_file: str = "test_results.json"):
        """Save timing records to JSON file"""
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, 'w') as f:
            json.dump({
                'test_summary': {
                    'total_iterations': len(self.timing_records),
                    'detected_count': sum(1 for r in self.timing_records if r['detected']),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'timing_records': self.timing_records
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def plot_results(self, output_file: str = "test_results.png"):
        """Plot timing results"""
        if not self.timing_records:
            print("No timing records to plot.")
            return
        
        iterations = [r['iteration'] for r in self.timing_records]
        server_times = [r['server_time'] for r in self.timing_records]
        network_times = [r['network_time'] for r in self.timing_records]
        detection_times = [r['detection_time'] for r in self.timing_records]
        projection_times = [r['vlm_projection_time'] for r in self.timing_records]
        inference_times = [r['vlm_inference_time'] for r in self.timing_records]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Server Performance Test Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Stacked bar chart of time breakdown
        ax1 = axes[0, 0]
        width = 0.8
        ax1.bar(iterations, detection_times, width, label='Detection', color='#FF6B6B')
        ax1.bar(iterations, projection_times, width, bottom=detection_times, 
                label='VLM Projection', color='#4ECDC4')
        
        bottom = np.array(detection_times) + np.array(projection_times)
        ax1.bar(iterations, inference_times, width, bottom=bottom, 
                label='VLM Inference', color='#45B7D1')
        
        bottom = bottom + np.array(inference_times)
        ax1.bar(iterations, network_times, width, bottom=bottom, 
                label='Network', color='#FFA07A')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Time Breakdown per Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total time per iteration
        ax2 = axes[0, 1]
        total_times = [r['total_time'] for r in self.timing_records]
        ax2.plot(iterations, total_times, marker='o', linewidth=2, markersize=6, color='#5A67D8')
        ax2.axhline(y=np.mean(total_times), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(total_times):.3f}s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Total Time per Iteration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Component time comparison
        ax3 = axes[1, 0]
        components = ['Detection', 'Projection', 'Inference', 'Network']
        means = [
            np.mean(detection_times),
            np.mean(projection_times),
            np.mean(inference_times),
            np.mean(network_times)
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax3.bar(components, means, color=colors)
        ax3.set_ylabel('Mean Time (seconds)')
        ax3.set_title('Average Time by Component')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s',
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Detection scores
        ax4 = axes[1, 1]
        scores = [r['detection_score'] for r in self.timing_records]
        detected = [r['detected'] for r in self.timing_records]
        colors_detected = ['green' if d else 'red' for d in detected]
        ax4.scatter(iterations, scores, c=colors_detected, s=100, alpha=0.6)
        ax4.axhline(y=0.5, color='gray', linestyle='--', label='Threshold (0.5)')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Detection Score')
        ax4.set_title('Detection Scores (Green=Detected, Red=Not Detected)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
        
        # plt.show()
    
    def run_test(self, num_iterations: int = 20, goal: str = "banana"):
        """Run complete test"""
        print("\n" + "="*80)
        print("SERVER PERFORMANCE TEST")
        print("="*80)
        print(f"Server URL: {self.server_url}")
        print(f"Iterations: {num_iterations}")
        print(f"Goal: {goal}")
        print("="*80)
        
        # Check server health
        if not self.test_health():
            print("\n✗ Server is not available. Please start the server first.")
            return False
        
        # Reset navigation
        try:
            self.reset_navigation(goal=goal, goal_description="Test object for performance testing")
        except Exception as e:
            print(f"\n✗ Failed to reset navigation: {e}")
            return False
        
        # Run iterations
        print(f"\n{'='*80}")
        print(f"Starting {num_iterations} iterations...")
        print(f"{'='*80}")
        
        for i in range(1, num_iterations + 1):
            try:
                self.send_navigation_step(i)
                # Small delay between iterations
                time.sleep(0.5)
            except Exception as e:
                print(f"\n✗ Iteration {i} failed: {e}")
                print("Continuing with next iteration...")
                continue
        
        # Print statistics
        self.print_statistics()
        
        # Save results
        self.save_results()
        
        # Plot results
        try:
            self.plot_results()
        except Exception as e:
            print(f"\nWarning: Failed to generate plots: {e}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Test navigation server performance')
    parser.add_argument('--server', type=str, default='http://10.19.126.158:1874',
                       help='Server URL (default: http://10.19.126.158:1874)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of test iterations (default: 20)')
    parser.add_argument('--goal', type=str, default='banana',
                       help='Navigation goal (default: banana)')
    
    args = parser.parse_args()
    
    tester = ServerTester(server_url=args.server)
    
    try:
        success = tester.run_test(num_iterations=args.iterations, goal=args.goal)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        if tester.timing_records:
            print("\nPartial results:")
            tester.print_statistics()
            tester.save_results()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

