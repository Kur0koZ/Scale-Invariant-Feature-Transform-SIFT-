import cv2
import numpy as np
import matplotlib.pyplot as plt
from function import detect_and_describe, match_descriptors, ransac_affine, visualize
import time


def main():

    obj = cv2.imread("One_Piece.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
    scene = cv2.imread("One_piece.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)

    kps_obj, desc_obj = detect_and_describe(obj, n_octaves=3)
    kps_scene, desc_scene = detect_and_describe(scene, n_octaves=3)
    print(f" object keypoints: {len(kps_obj)} | scene keypoints: {len(kps_scene)}")

    # print("\nSTEP 3: Matching + Lowe's Ratio Test")

    matches = match_descriptors(desc_obj, desc_scene, ratio_thresh=0.75)
    print(f" match ที่ผ่าน ratio test: {len(matches)} / {len(kps_obj)}")
    #for ratio in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:

    for m in matches:
        print(f"   obj#{m[0]} <-> scene#{m[1]}  d1={m[2]:.4f}  ratio={m[3]:.4f}")

    if len(matches) < 3:
        print(
            "   match ไม่พอสำหรับ RANSAC (ต้องการอย่างน้อย 3 คู่) - ลด ratio_thresh หรือเพิ่ม keypoint"
        )
        return kps_obj, kps_scene, matches, None, None, None, obj, scene

    src_pts = np.array([[kps_obj[m[0]]["x"], kps_obj[m[0]]["y"]] for m in matches])
    dst_pts = np.array([[kps_scene[m[1]]["x"], kps_scene[m[1]]["y"]] for m in matches])

    print("\nSTEP 4: RANSAC หา affine transform")

    M, t, inlier_idx = ransac_affine(src_pts, dst_pts, n_iter=3000, threshold=8.0)
    if M is None:
        print("   RANSAC หา transform ที่ inlier พอไม่ได้")
        return kps_obj, kps_scene, matches, None, None, None, obj, scene
    scale_est = np.sqrt(abs(np.linalg.det(M)))
    theta_est = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    h, w = obj.shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    scene_box = (M @ corners.T).T + t
    for i, pt in enumerate(scene_box):
        print(f"   มุมที่ {i+1}: ({pt[0]:.1f},{pt[1]:.1f})")

if __name__ == "__main__":
    main()