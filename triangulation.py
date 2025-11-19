"""
Multi-view 3D point triangulation using DLT
"""
import numpy as np
import cv2

class Triangulator:
    """
    Multi-view triangulation using Direct Linear Transform (DLT)
    
    Theory:
    - Given: 2D point correspondences across N views, camera projection matrices
    - Goal: Recover 3D point X
    - Method: Each view gives equation x = PX, reformulate as homogeneous linear system AX = 0
    """
    
    def __init__(self, projection_matrices):
        """
        Initialize triangulator with camera projection matrices
        
        Args:
            projection_matrices: List of 3x4 or 4x4 projection matrices [P1, P2, ...]
                                P maps 3D homogeneous coords → 2D homogeneous coords
        """
        self.P_matrices = [np.array(P) for P in projection_matrices]
        self.num_cameras = len(self.P_matrices)
        
        # Ensure 3x4 format
        for i, P in enumerate(self.P_matrices):
            if P.shape == (4, 4):
                self.P_matrices[i] = P[:3, :]
            assert self.P_matrices[i].shape == (3, 4), f"P{i} must be 3x4, got {P.shape}"
    
    def triangulate_dlt(self, points_2d):
        """
        Triangulate single 3D point from 2D correspondences using DLT
        
        Math:
        For each view i with point (u_i, v_i) and projection matrix P_i:
            [u_i]       [X]
            [v_i] = P_i [Y]
            [1  ]       [Z]
                        [1]
        
        Cross-product formulation eliminates scale:
            u_i = (P_i[0]·X) / (P_i[2]·X)  →  u_i(P_i[2]·X) - (P_i[0]·X) = 0
            v_i = (P_i[1]·X) / (P_i[2]·X)  →  v_i(P_i[2]·X) - (P_i[1]·X) = 0
        
        Stack to form A·X = 0, solve via SVD
        
        Args:
            points_2d: List of (u, v) coordinates, one per camera [(u1,v1), (u2,v2), ...]
        
        Returns:
            point_3d: (X, Y, Z) 3D point in world coordinates or None if invalid
        """
        assert len(points_2d) >= 2, "Need at least 2 views for triangulation"
        assert len(points_2d) == self.num_cameras, f"Expected {self.num_cameras} points, got {len(points_2d)}"
        
        A = []
        for i, (u, v) in enumerate(points_2d):
            P = self.P_matrices[i]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        
        A = np.array(A)
        
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        
        X = X_homogeneous[:3] / X_homogeneous[3]
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return None
        
        return X
    
    def triangulate_opencv(self, points_2d):
        """
        Triangulate using OpenCV's built-in method (for comparison/validation)
        Currently only supports 2-view triangulation
        
        Args:
            points_2d: List of (u, v) coordinates for 2 cameras
            
        Returns:
            point_3d: (X, Y, Z) 3D point
        """
        assert len(points_2d) == 2, "OpenCV method requires exactly 2 views"
        
        pts1 = np.array(points_2d[0], dtype=np.float32).reshape(1, 2)
        pts2 = np.array(points_2d[1], dtype=np.float32).reshape(1, 2)
        
        # OpenCV expects points as 2xN arrays
        pts1 = pts1.T
        pts2 = pts2.T
        
        points_4d = cv2.triangulatePoints(self.P_matrices[0], self.P_matrices[1], pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous
        
        return points_3d.flatten()

    def is_in_front_of_camera(self, point_3d, cam_idx):
        """
        Check if 3D point has positive depth (in front of camera)
        
        Args:
            point_3d: (X, Y, Z) 3D point
            cam_idx: Camera index
        
        Returns:
            True if point is in front of camera
        """
        P = self.P_matrices[cam_idx]
        X_hom = np.append(point_3d, 1)
        depth = P[2] @ X_hom
        return depth > 0
    
    def triangulate_points(self, cam_idx1, cam_idx2, points1, points2):
        """
        Triangulate a batch of points from two views with depth validation.
        """
        P1 = self.P_matrices[cam_idx1]
        P2 = self.P_matrices[cam_idx2]

        points1 = np.array(points1, dtype=np.float32).T
        points2 = np.array(points2, dtype=np.float32).T

        points_4d_hom = cv2.triangulatePoints(P1, P2, points1, points2)
        
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        points_3d = points_3d.T
        
        valid_points = []
        for pt in points_3d:
            if self.is_in_front_of_camera(pt, cam_idx1) and self.is_in_front_of_camera(pt, cam_idx2):
                if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
                    valid_points.append(pt)
        
        return np.array(valid_points)
    
    def triangulate_batch(self, points_2d_batch, method='dlt'):
        """
        Triangulate multiple 3D points from batches of 2D correspondences
        
        Args:
            points_2d_batch: List of correspondence sets
                           [[(u1,v1), (u2,v2), ...],  # Point 1 in all views
                            [(u1,v1), (u2,v2), ...],  # Point 2 in all views
                            ...]
            method: 'dlt' or 'opencv'
        
        Returns:
            points_3d: Nx3 array of 3D points
        """
        points_3d = []
        
        for correspondences in points_2d_batch:
            try:
                if method == 'dlt':
                    pt_3d = self.triangulate_dlt(correspondences)
                elif method == 'opencv':
                    pt_3d = self.triangulate_opencv(correspondences)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if pt_3d is None:
                    continue
                
                if all(self.is_in_front_of_camera(pt_3d, i) for i in range(self.num_cameras)):
                    points_3d.append(pt_3d)
            except Exception as e:
                continue
        
        return np.array(points_3d) if points_3d else np.array([])
    
    def compute_reprojection_error(self, point_3d, points_2d):
        """
        Compute reprojection error to validate triangulation quality
        
        Theory: Project 3D point back to 2D, measure pixel distance from original
        Good triangulation → small error (<1-2 pixels)
        
        Args:
            point_3d: (X, Y, Z) 3D point
            points_2d: List of original 2D observations [(u1,v1), (u2,v2), ...]
        
        Returns:
            errors: List of reprojection errors (one per view)
            mean_error: Average reprojection error across all views
        """
        X_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coords
        
        errors = []
        for i, (u_obs, v_obs) in enumerate(points_2d):
            # Project 3D point to 2D using projection matrix
            x_proj_hom = self.P_matrices[i] @ X_homogeneous
            u_proj = x_proj_hom[0] / x_proj_hom[2]
            v_proj = x_proj_hom[1] / x_proj_hom[2]
            
            # Euclidean distance in image plane
            error = np.sqrt((u_obs - u_proj)**2 + (v_obs - v_proj)**2)
            errors.append(error)
        
        return errors, np.mean(errors)
    
    def filter_by_reprojection_error(self, points_3d, points_2d_batch, threshold=2.0):
        """
        Filter 3D points based on reprojection error
        
        Args:
            points_3d: Nx3 array of 3D points
            points_2d_batch: List of correspondence sets matching points_3d
            threshold: Maximum allowed reprojection error in pixels
        
        Returns:
            filtered_points_3d: Filtered 3D points
            filtered_indices: Indices of points that passed filter
            errors: Mean reprojection errors for all points
        """
        filtered_points = []
        filtered_indices = []
        all_errors = []
        
        for i, (pt_3d, pts_2d) in enumerate(zip(points_3d, points_2d_batch)):
            _, mean_error = self.compute_reprojection_error(pt_3d, pts_2d)
            all_errors.append(mean_error)
            
            if mean_error < threshold:
                filtered_points.append(pt_3d)
                filtered_indices.append(i)
        
        print(f"Filtered {len(points_3d)} → {len(filtered_points)} points (threshold={threshold}px)")
        print(f"Mean error: {np.mean(all_errors):.2f}px, Max: {np.max(all_errors):.2f}px")
        
        return np.array(filtered_points), filtered_indices, all_errors


def create_projection_matrix(K, R, t):
    """
    Helper: Create projection matrix P = K[R|t]
    
    Args:
        K: 3x3 intrinsic matrix (from calibration)
        R: 3x3 rotation matrix (camera orientation)
        t: 3x1 translation vector (camera position)
    
    Returns:
        P: 3x4 projection matrix
    """
    # Ensure t is column vector
    t = np.array(t).reshape(3, 1)
    
    # P = K[R|t]
    Rt = np.hstack([R, t])
    P = K @ Rt
    
    return P


# ===== SYNTHETIC TESTS FOR VALIDATION =====

def test_triangulation_synthetic():
    """
    Validate triangulation with synthetic data where ground truth is known
    """
    print("=" * 60)
    print("SYNTHETIC TRIANGULATION TEST")
    print("=" * 60)
    
    # Ground truth 3D point
    X_true = np.array([1.5, 2.0, 5.0])
    print(f"\nGround truth 3D point: {X_true}")
    
    # Camera 1: At origin looking down +Z axis
    K1 = np.array([[800, 0, 320],
                   [0, 800, 240],
                   [0, 0, 1]], dtype=float)
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = create_projection_matrix(K1, R1, t1)
    
    # Camera 2: Translated 1m to the right (baseline)
    K2 = K1.copy()
    R2 = np.eye(3)
    t2 = np.array([[1.0], [0], [0]])  # 1m baseline
    P2 = create_projection_matrix(K2, R2, t2)
    
    # Project ground truth point to both cameras
    X_hom = np.append(X_true, 1)
    
    x1_hom = P1 @ X_hom
    u1, v1 = x1_hom[0] / x1_hom[2], x1_hom[1] / x1_hom[2]
    
    x2_hom = P2 @ X_hom
    u2, v2 = x2_hom[0] / x2_hom[2], x2_hom[1] / x2_hom[2]
    
    print(f"Projected to camera 1: ({u1:.2f}, {v1:.2f})")
    print(f"Projected to camera 2: ({u2:.2f}, {v2:.2f})")
    
    # Triangulate back to 3D
    triangulator = Triangulator([P1, P2])
    
    # Test DLT
    X_dlt = triangulator.triangulate_dlt([(u1, v1), (u2, v2)])
    error_dlt = np.linalg.norm(X_dlt - X_true)
    print(f"\nDLT reconstruction: {X_dlt}")
    print(f"DLT error: {error_dlt:.6f} meters")
    
    # Test OpenCV
    X_opencv = triangulator.triangulate_opencv([(u1, v1), (u2, v2)])
    error_opencv = np.linalg.norm(X_opencv - X_true)
    print(f"\nOpenCV reconstruction: {X_opencv}")
    print(f"OpenCV error: {error_opencv:.6f} meters")
    
    # Test reprojection error
    errors, mean_error = triangulator.compute_reprojection_error(X_dlt, [(u1, v1), (u2, v2)])
    print(f"\nReprojection errors: {errors}")
    print(f"Mean reprojection error: {mean_error:.6f} pixels")
    
    # Test with noise
    print("\n" + "-" * 60)
    print("Testing with noisy observations (±0.5 pixel)")
    noise = np.random.randn(2, 2) * 0.5  # Small pixel noise
    u1_noisy, v1_noisy = u1 + noise[0, 0], v1 + noise[0, 1]
    u2_noisy, v2_noisy = u2 + noise[1, 0], v2 + noise[1, 1]
    
    X_noisy = triangulator.triangulate_dlt([(u1_noisy, v1_noisy), (u2_noisy, v2_noisy)])
    error_noisy = np.linalg.norm(X_noisy - X_true)
    print(f"Noisy reconstruction: {X_noisy}")
    print(f"Error with noise: {error_noisy:.6f} meters")
    
    # Test batch triangulation
    print("\n" + "-" * 60)
    print("Testing batch triangulation (5 points)")
    
    # Generate 5 random 3D points
    points_3d_true = np.random.rand(5, 3) * 5 + np.array([0, 0, 3])  # Z between 3-8m
    
    # Project to both cameras
    points_2d_batch = []
    for pt in points_3d_true:
        pt_hom = np.append(pt, 1)
        
        x1 = P1 @ pt_hom
        x2 = P2 @ pt_hom
        
        u1, v1 = x1[0] / x1[2], x1[1] / x1[2]
        u2, v2 = x2[0] / x2[2], x2[1] / x2[2]
        
        points_2d_batch.append([(u1, v1), (u2, v2)])
    
    # Triangulate batch
    points_3d_reconstructed = triangulator.triangulate_batch(points_2d_batch)
    
    # Compute errors
    errors = np.linalg.norm(points_3d_reconstructed - points_3d_true, axis=1)
    print(f"Batch reconstruction errors: {errors}")
    print(f"Mean error: {np.mean(errors):.6f} meters")
    print(f"Max error: {np.max(errors):.6f} meters")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!" if np.mean(errors) < 1e-6 else "✗ Tests failed")
    print("=" * 60)


if __name__ == "__main__":
    test_triangulation_synthetic()