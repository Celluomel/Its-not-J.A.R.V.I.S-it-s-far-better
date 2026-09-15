"""Enhanced test script to diagnose camera issues with timeout protection"""
import cv2
import platform
import threading
import time
import subprocess
import sys

print(f"Platform: {platform.system()} {platform.release()}")
print(f"OpenCV version: {cv2.__version__}")
print()

def test_camera_with_timeout(camera_id, backend, timeout=5):
    """Test camera with timeout protection"""
    result = {"success": False, "frame": None, "error": None}
    
    def _test():
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                # Give camera time to initialize
                time.sleep(0.5)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    result["success"] = True
                    result["frame"] = frame
                else:
                    result["error"] = "Failed to read frame"
            else:
                result["error"] = "Failed to open"
        except Exception as e:
            result["error"] = str(e)
    
    thread = threading.Thread(target=_test)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return {"success": False, "error": "Timeout - camera not responding"}
    return result

# First, check if camera is being used by another process on Windows
if platform.system() == "Windows":
    print("Checking for processes that might be using the camera...")
    try:
        # Check for common apps that might use camera
        result = subprocess.run(
            ['powershell', '-Command', 'Get-Process | Where-Object { $_.ProcessName -match "zoom|teams|skype|slack|chrome|firefox|edge|camera|obs" } | Select-Object ProcessName'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.stdout.strip():
            print("Found these processes that might be using the camera:")
            print(result.stdout)
            print("Try closing these applications and run the test again.")
        else:
            print("No obvious camera-using processes found.")
    except Exception as e:
        print(f"Could not check processes: {e}")
    print()

# Test different camera indices with timeouts
for camera_id in [0, 1, 2]:
    print(f"Testing camera {camera_id}...")
    
    # Try different backends
    backends = []
    if platform.system() == "Windows":
        backends = [
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_ANY, "ANY")
        ]
    else:
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "ANY")
        ]
    
    for backend, backend_name in backends:
        print(f"  Trying backend {backend_name}...", end=" ")
        sys.stdout.flush()
        
        result = test_camera_with_timeout(camera_id, backend, timeout=3)
        
        if result["success"]:
            frame = result["frame"]
            print(f"✅ SUCCESS - Frame shape: {frame.shape}")
            # Save a test frame
            cv2.imwrite(f"test_frame_cam{camera_id}.jpg", frame)
            print(f"  Saved test_frame_cam{camera_id}.jpg")
            break
        else:
            print(f"❌ {result['error']}")
    
    print()

# Try DirectShow with extended parameters on Windows
if platform.system() == "Windows":
    print("Testing DirectShow with MJPG format...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Try to set MJPG format
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            time.sleep(0.5)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("✅ MJPG format test successful!")
                cv2.imwrite("test_frame_mjpg.jpg", frame)
            else:
                print("❌ MJPG format test failed")
        else:
            print("❌ Could not open camera with MJPG settings")
    except Exception as e:
        print(f"❌ MJPG test error: {e}")

print("\n" + "="*50)
print("TROUBLESHOOTING STEPS:")
print("="*50)
print("\n1. Close all applications that might use the camera:")
print("   - Zoom, Teams, Skype, Discord")
print("   - Chrome/Firefox (close all browser tabs)")
print("   - Camera app, OBS, Any video software")
print("\n2. Check Windows Camera Privacy Settings:")
print("   - Press Windows + I to open Settings")
print("   - Go to Privacy & Security → Camera")
print("   - Ensure 'Camera access' is ON")
print("   - Ensure apps have permission")
print("\n3. Restart the Windows Camera Service:")
print("   - Press Windows + R, type 'services.msc'")
print("   - Find 'Windows Camera Frame Server'")
print("   - Right-click and select 'Restart'")
print("\n4. Update camera drivers:")
print("   - Press Windows + X, select 'Device Manager'")
print("   - Expand 'Cameras' or 'Imaging devices'")
print("   - Right-click your camera and select 'Update driver'")
print("\n5. If using a laptop, check for a physical camera switch")
print("\n6. Restart your computer and try again")