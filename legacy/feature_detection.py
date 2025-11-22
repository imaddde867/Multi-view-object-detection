import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, num_features=3000, lowe_ratio=0.7, quality_level=0.1):
        self.orb = cv2.ORB_create(nfeatures=num_features, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.lowe_ratio = lowe_ratio
        self.quality_level = quality_level

    def detect_features(self, img):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(kp) == 0:
            return [], None
            
        return kp, des

    def match_features(self, des1, des2, kp1, kp2):
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return []
        
        try:
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:200]
            
            if len(matches) < 8:
                return []
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 5.0)

            if F is None or mask is None:
                return []
            
            inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
            
            return inlier_matches if len(inlier_matches) >= 8 else []
        except:
            return []

def draw_matches(img1, kp1, img2, kp2, matches):
    """
    Draw matches between two images.
    """
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


if __name__ == '__main__':
    try:
        img1_path = 'multiclass_ground_truth_images/c0/00000001.jpg'
        img2_path = 'multiclass_ground_truth_images/c1/00000001.jpg'
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise FileNotFoundError("Could not load images. Check file paths.")

        detector = FeatureDetector()
        
        kp1, des1 = detector.detect_features(img1)
        kp2, des2 = detector.detect_features(img2)
        
        matches = detector.match_features(des1, des2, kp1, kp2)

        print(f"Found {len(matches)} good matches.")

        matched_img = draw_matches(img1, kp1, img2, kp2, matches)

        cv2.imshow('Matches', matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except (FileNotFoundError, cv2.error) as e:
        print(f"Error: {e}")
        print("Please ensure that the image paths are correct and OpenCV is installed correctly.")
        print("Example paths are for the EPFL dataset structure.")