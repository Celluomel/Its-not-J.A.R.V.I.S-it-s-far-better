"""Ultimate Camera Test - Tries every possible combination"""
import cv2
import numpy as np
import time
import os
import platform
import subprocess
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def test_camera_combinations():
    """Test all possible camera combinations"""
    
    print_header("ULTIMATE CAMERA TEST")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Step 1: Check if camera is being used by another process
    print_header("STEP 1: Checking for processes using camera")
    try:
        result = subprocess.run(
            ['powershell', '-Command', 
             'Get-Process | Where-Object { $_.ProcessName -match "zoom|teams|skype|slack|chrome|firefox|edge|camera|obs|discord|whatsapp|telegram" } | Select-Object ProcessName, Id'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            print("Found these processes that might be using the camera:")
            print(result.stdout)
            print("\n⚠️  Close these applications and try again!")
        else:
            print("✅ No obvious camera-using processes found")
    except Exception as e:
        print(f"Could not check processes: {e}")
    
    # Step 2: Try to reset camera service
    print_header("STEP 2: Quick camera service reset")
    print("To reset camera service, run PowerShell as Administrator:")
    print("  net stop FrameServer")
    print("  net start FrameServer")
    
    # Step 3: Test all combinations
    print_header("STEP 3: Testing all camera combinations")
    
    working_combinations = []
    
    # Test cameras 0-2
    for cam_id in [0, 1, 2]:
        print(f"\n📷 Testing camera {cam_id}...")
        
        # Test different backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (DSHOW)"),
            (cv2.CAP_MSMF, "Media Foundation (MSMF)"),
            (cv2.CAP_ANY, "Auto (ANY)"),
        ]
        
        for backend, backend_name in backends:
            print(f"\n  🔧 Trying {backend_name}...")
            
            # Try different resolutions
            resolutions = [
                (640, 480, "640x480"),
                (320, 240, "320x240"),
                (1280, 720, "720p"),
                (0, 0, "Default"),
            ]
            
            for width, height, res_name in resolutions:
                print(f"    📏 Resolution: {res_name}")
                
                try:
                    # Open camera
                    cap = cv2.VideoCapture(cam_id, backend)
                    
                    if not cap.isOpened():
                        print(f"      ❌ Could not open camera")
                        cap.release()
                        continue
                    
                    # Set resolution if specified
                    if width > 0 and height > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Try different formats
                    formats = [
                        (cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), "MJPG"),
                        (cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'), "YUYV"),
                        (0, "Default"),
                    ]
                    
                    for fourcc, format_name in formats:
                        print(f"      🎨 Format: {format_name}")
                        
                        if fourcc != 0:
                            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                        
                        # Try to read a frame with timeout
                        success = False
                        frame = None
                        
                        for attempt in range(3):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                success = True
                                break
                            time.sleep(0.5)
                        
                        if success:
                            print(f"        ✅ SUCCESS! Frame shape: {frame.shape}")
                            
                            # Save the frame
                            filename = f"camera{cam_id}_{backend_name[:4]}_{format_name}_{res_name}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"        💾 Saved: {filename}")
                            
                            working_combinations.append({
                                'camera': cam_id,
                                'backend': backend_name,
                                'format': format_name,
                                'resolution': res_name
                            })
                            
                            # If we found a working combination, we can stop testing this camera
                            break
                        else:
                            print(f"        ❌ Could not read frame")
                    
                    cap.release()
                    
                    if working_combinations:
                        break
                        
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    try:
                        cap.release()
                    except:
                        pass
            
            if working_combinations:
                break
        
        if working_combinations:
            break
    
    return working_combinations

def test_windows_camera_app():
    """Guide user to test Windows Camera app"""
    print_header("STEP 4: Testing Windows Camera App")
    print("1. Press Windows key, type 'Camera' and open the Camera app")
    print("2. Check if you can see yourself and take a photo")
    print("3. If it works, close the Camera app completely")
    print("4. Press Enter to continue...")
    input()

def main():
    """Main test function"""
    
    # Test all combinations
    working = test_camera_combinations()
    
    if working:
        print_header("✅ SUCCESS! Working configurations found:")
        for combo in working:
            print(f"  Camera {combo['camera']} with {combo['backend']}, {combo['format']}, {combo['resolution']}")
        
        print("\n📝 Update your vision_manager.py with:")
        print("""
    def _get_camera_backends(self):
        if platform.system() == "Windows":
            return [
                cv2.CAP_MSMF,  # Your working backend
                cv2.CAP_DSHOW,
                cv2.CAP_ANY,
            ]
        """)
    else:
        print_header("❌ NO WORKING CAMERA FOUND")
        print("\nLet's fix this step by step:")
        
        print("\n1️⃣  Check Windows Camera App:")
        print("   - Open Windows Camera app")
        print("   - Does it show video?")
        print("   - If NO: Your camera hardware/drivers have issues")
        print("   - If YES: Close it and try running this test again")
        
        print("\n2️⃣  Reset Windows Camera Service:")
        print("   - Open PowerShell as Administrator")
        print("   - Run these commands:")
        print("     net stop FrameServer")
        print("     net start FrameServer")
        
        print("\n3️⃣  Update Camera Drivers:")
        print("   - Open Device Manager (Win + X → Device Manager)")
        print("   - Expand 'Cameras' or 'Imaging devices'")
        print("   - Right-click your camera → Update driver")
        print("   - Choose 'Search automatically for drivers'")
        
        print("\n4️⃣  Check Windows Privacy Settings:")
        print("   - Settings → Privacy & Security → Camera")
        print("   - Ensure 'Camera access' is ON")
        print("   - Ensure 'Let apps access your camera' is ON")
        
        print("\n5️⃣  Disable and re-enable camera:")
        print("   - In Device Manager, right-click camera")
        print("   - Select 'Disable device'")
        print("   - Wait 10 seconds")
        print("   - Right-click → 'Enable device'")
        
        print("\n6️⃣  Restart your computer")
        
        print("\n7️⃣  If still not working, try a USB camera or check if built-in camera is faulty")

if __name__ == "__main__":
    main()