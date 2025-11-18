"""
camera_calibration.py - Calibrate cameras and compute projection matrices
"""
import numpy as np
import cv2
import json
import os

class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Initialize camera calibrator
        
        Args:
            checkerboard_size: (cols, rows) of internal corners in checkerboard
            square_size: Size of checkerboard squares in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
    def calibrate_single_camera(self, images):
        """
        Calibrate a single camera from checkerboard images
        
        Args:
            images: List of images containing checkerboard pattern
            
        Returns:
            ret: Calibration success flag
            camera_matrix: 3x3 intrinsic matrix K
            dist_coeffs: Distortion coefficients
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in image plane
        
        print(f"Calibrating camera with {len(images)} images...")
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                objpoints.append(self.objp)
                
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                print(f"  ✓ Image {i+1}/{len(images)} - corners found")
            else:
                print(f"  ✗ Image {i+1}/{len(images)} - corners not found")
        
        if len(objpoints) == 0:
            raise ValueError("No checkerboard patterns found in images!")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        print(f"\nCalibration {'successful' if ret else 'failed'}!")
        print(f"Reprojection error: {ret:.4f}")
        
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs
    
    def stereo_calibrate(self, images_left, images_right):
        """
        Perform stereo calibration between two cameras
        
        Args:
            images_left: List of images from left camera
            images_right: List of images from right camera
            
        Returns:
            Dictionary containing all calibration parameters
        """
        print("\n=== STEREO CALIBRATION ===\n")
        
        # Calibrate individual cameras
        print("Calibrating LEFT camera...")
        ret_l, K_l, dist_l, rvecs_l, tvecs_l = self.calibrate_single_camera(images_left)
        
        print("\nCalibrating RIGHT camera...")
        ret_r, K_r, dist_r, rvecs_r, tvecs_r = self.calibrate_single_camera(images_right)
        
        # Find common checkerboard points
        objpoints = []
        imgpoints_l = []
        imgpoints_r = []
        
        print("\nFinding corresponding points...")
        for img_l, img_r in zip(images_left, images_right):
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.checkerboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.checkerboard_size, None)
            
            if ret_l and ret_r:
                objpoints.append(self.objp)
                
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
        
        # Stereo calibration
        print(f"\nPerforming stereo calibration with {len(objpoints)} image pairs...")
        
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            K_l, dist_l, K_r, dist_r,
            gray_l.shape[::-1], criteria=criteria, flags=flags
        )
        
        print(f"\nStereo calibration error: {ret:.4f}")
        print(f"Baseline distance: {np.linalg.norm(T):.4f} meters")
        
        # Compute projection matrices
        # Left camera is at origin
        R_l = np.eye(3)
        t_l = np.zeros((3, 1))
        P_l = K_l @ np.hstack([R_l, t_l])
        
        # Right camera transformation
        R_r = R
        t_r = T
        P_r = K_r @ np.hstack([R_r, t_r])
        
        calibration_data = {
            'K_left': K_l.tolist(),
            'dist_left': dist_l.tolist(),
            'K_right': K_r.tolist(),
            'dist_right': dist_r.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'P_left': P_l.tolist(),
            'P_right': P_r.tolist(),
            'baseline': float(np.linalg.norm(T)),
            'stereo_error': float(ret)
        }
        
        return calibration_data
    
    def save_calibration(self, calibration_data, filename='calibration.json'):
        """Save calibration data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"\n✓ Calibration saved to {filename}")
    
    def load_calibration(self, filename='calibration.json'):
        """Load calibration data from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        for key in ['K_left', 'dist_left', 'K_right', 'dist_right', 'R', 'T', 'E', 'F', 'P_left', 'P_right']:
            if key in data:
                data[key] = np.array(data[key])
        
        return data


def capture_calibration_images(camera_id=0, num_images=20, save_dir='calibration_images'):
    """
    Interactive tool to capture calibration images
    
    Args:
        camera_id: Camera device ID
        num_images: Number of images to capture
        save_dir: Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    count = 0
    
    print(f"\n=== CALIBRATION IMAGE CAPTURE ===")
    print(f"Target: {num_images} images")
    print("Controls:")
    print("  SPACE - Capture image")
    print("  Q - Quit")
    print("\nMove the checkerboard to different positions and angles")
    print("Make sure the entire checkerboard is visible\n")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        cv2.putText(display, f"Captured: {count}/{num_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE: Capture | Q: Quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Calibration Capture', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = os.path.join(save_dir, f'calib_{count:03d}.jpg')
            cv2.imwrite(filename, frame)
            print(f"✓ Captured image {count + 1}/{num_images}")
            count += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Captured {count} images in {save_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'capture':
        # Capture calibration images
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        capture_calibration_images(camera_id)
    else:
        print("Camera Calibration Module")
        print("=" * 50)
        print("\nUsage:")
        print("  python camera_calibration.py capture [camera_id]")
        print("\nExample:")
        print("  python camera_calibration.py capture 0")
