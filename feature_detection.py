import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, num_features=5000, lowe_ratio=0.8):
        self.orb = cv2.ORB_create(nfeatures=num_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.lowe_ratio = lowe_ratio

    def detect_features(self, img):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match_features(self, des1, des2, kp1, kp2):
        matches = self.bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        if matches:
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.lowe_ratio * n.distance:
                        good_matches.append(m)

        if len(good_matches) > 7:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 5.0)

            if mask is not None:
                inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
                return inlier_matches
        
        return good_matches

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