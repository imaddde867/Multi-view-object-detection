import json
import numpy as np

def create_epfl_calibration():
    # These are the standard Projection Matrices (P) for the EPFL "Terrace" dataset
    # Format: P = K * [R | t]
    calibration_data = {
        "cameras": {
            "camera0": {
                "P": [
                    [1.138740e+03, 1.163200e+02, -3.004400e+02, -1.709100e+06],
                    [-2.388800e+01, 8.033850e+02, -9.092780e+02, 4.846030e+06],
                    [-7.260000e-02, 9.907200e-01, 1.264600e-01, 4.199480e+03]
                ]
            },
            "camera1": {
                "P": [
                    [1.143210e+03, -2.299200e+02, 1.912600e+02, -4.098180e+06],
                    [1.276500e+02, 7.302700e+02, -9.832800e+02, 5.193580e+06],
                    [1.358000e-01, 9.885000e-01, 7.378000e-02, 4.093180e+03]
                ]
            },
            "camera2": {
                 "P": [
                    [1.047900e+03, -5.572600e+02, 3.861400e+02, -6.193440e+06],
                    [2.866600e+02, 5.857000e+02, -1.013040e+03, 5.337840e+06],
                    [3.727000e-01, 9.266800e-01, 5.260000e-02, 4.010440e+03]
                 ]
            },
            "camera3": {
                "P": [
                    [8.728400e+02, -7.693000e+02, 4.523400e+02, -7.337210e+06],
                    [4.258200e+02, 4.276500e+02, -1.015360e+03, 5.334780e+06],
                    [5.739000e-01, 8.185200e-01, 2.500000e-02, 4.148700e+03]
                ]
            }
        }
    }

    file_path = 'epfl-calibration.json'
    with open(file_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"✓ Successfully created '{file_path}'")
    print("  You can now run main.py")

if __name__ == "__main__":
    create_epfl_calibration()