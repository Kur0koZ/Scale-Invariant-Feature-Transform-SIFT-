import numpy as np
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import numpy as np


def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    # Normalize ให้ผลรวมของ kernel = 1
    kernel /= np.sum(kernel)

    return kernel

def convolution2d(img, kernel):
    k = kernel.shape[0]
    pad = k // 2
    # Reflect padding
    padded = np.pad(img, pad, mode="reflect")
    output = np.zeros(img.shape, dtype=np.float64)

    h, w = img.shape

    for y in range(h):
        for x in range(w):
            region = padded[y : y + k, x : x + k]
            output[y, x] = np.sum(region * kernel)

    return output


def gaussian_blur(img, sigma):
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(size, sigma)
    return convolution2d(img, kernel)


def build_octave(img, sigma0, s, n_extra=3):

    k = 2 ** (1.0 / s)

    n = s + n_extra

    sigmas = [sigma0 * (k**i) for i in range(n)]

    Ls = [gaussian_blur(img, sigma=sig) for sig in sigmas]

    return sigmas, Ls


def make_object_image(size=128):
    img = Image.new("L", (size, size), color=30)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 108, 108], fill=200, outline=255, width=3)
    d.ellipse([48, 48, 80, 80], fill=90, outline=255, width=2)
    d.line([20, 20, 108, 108], fill=255, width=2)
    d.line([108, 20, 20, 108], fill=255, width=2)
    return np.array(img, dtype=np.float64)


def make_scene_image(obj_arr, size=320, angle=20, scale=0.9, tx=180, ty=140, seed=7):
    rng = np.random.default_rng(seed)
    bg = rng.integers(60, 110, (size, size)).astype(np.float64)
    scene = Image.fromarray(bg.astype(np.uint8), mode="L")
    d = ImageDraw.Draw(scene)
    for _ in range(4):
        cx, cy = rng.integers(30, size - 30, 2)
        r = int(rng.integers(12, 28))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.integers(120, 170)))

    obj_img = Image.fromarray(obj_arr.astype(np.uint8), mode="L")
    obj_img = obj_img.resize((int(obj_img.width * scale), int(obj_img.height * scale)))
    obj_img = obj_img.rotate(angle, expand=True, fillcolor=0)
    ow, oh = obj_img.size
    paste_x, paste_y = tx - ow // 2, ty - oh // 2

    scene_arr = np.array(scene, dtype=np.float64)
    obj_np = np.array(obj_img, dtype=np.float64)
    mask = obj_np > 5
    ys, xs = np.where(mask)
    scene_arr[ys + paste_y, xs + paste_x] = obj_np[ys, xs]
    return scene_arr, (paste_x, paste_y, ow, oh, angle, scale)


def build_pyramid(img, n_octaves=3, sigma0=1, s=3):

    octaves = []
    cur_img = img.astype(np.float64)
    factor = 1.0
    for o in range(n_octaves):
        sigmas, Ls = build_octave(cur_img, sigma0, s)
        DoGs = [Ls[i + 1] - Ls[i] for i in range(len(Ls) - 1)]
        octaves.append({"sigmas": sigmas, "Ls": Ls, "DoGs": DoGs, "factor": factor})
        # ภาพสำหรับ octave ถัดไป = downsample ภาพที่ blur ระดับ s (index s) ลงครึ่งหนึ่ง
        base = Ls[s]
        cur_img = base[::2, ::2]
        factor *= 2.0
        if min(cur_img.shape) < 8:
            break
    return octaves


def find_extrema(octaves, contrast_thresh=0.01, r_edge=10, img_range=255.0, border=2):
    keypoints = [] # each: dict x,y (ในพิกัดภาพต้นฉบับ), sigma, octave_idx, response
    edge_thresh = (r_edge + 1) ** 2 / r_edge
    for oi, oc in enumerate(octaves):
        DoGs = oc["DoGs"]
        sigmas = oc["sigmas"]
        factor = oc["factor"]
        H, W = DoGs[0].shape
        for si in range(1, len(DoGs) - 1):
            below, curr, above = DoGs[si - 1], DoGs[si], DoGs[si + 1]
            for r in range(border, H - border):
                for c in range(border, W - border):
                    val = curr[r, c]
                    if abs(val) < 0.5 * contrast_thresh * img_range:
                        continue # early skip เพื่อความเร็ว
                    patch = np.concatenate(
                        [
                            below[r - 1 : r + 2, c - 1 : c + 2].flatten(),
                            np.delete(curr[r - 1 : r + 2, c - 1 : c + 2].flatten(), 4),
                            above[r - 1 : r + 2, c - 1 : c + 2].flatten(),
                        ]
                    )
                    is_max = val > patch.max()
                    is_min = val < patch.min()
                    if not (is_max or is_min):
                        continue
                    # --- sub-pixel refine ---
                    Dx = (curr[r, c + 1] - curr[r, c - 1]) / 2
                    Dy = (curr[r + 1, c] - curr[r - 1, c]) / 2
                    Ds = (above[r, c] - below[r, c]) / 2
                    grad = np.array([Dx, Dy, Ds])
                    Dxx = curr[r, c + 1] - 2 * curr[r, c] + curr[r, c - 1]
                    Dyy = curr[r + 1, c] - 2 * curr[r, c] + curr[r - 1, c]
                    Dss = above[r, c] - 2 * curr[r, c] + below[r, c]
                    Dxy = (
                        curr[r + 1, c + 1]
                        - curr[r + 1, c - 1]
                        - curr[r - 1, c + 1]
                        + curr[r - 1, c - 1]
                    ) / 4
                    Dxs = (
                        above[r, c + 1]
                        - above[r, c - 1]
                        - below[r, c + 1]
                        + below[r, c - 1]
                    ) / 4
                    Dys = (
                        above[r + 1, c]
                        - above[r - 1, c]
                        - below[r + 1, c]
                        + below[r - 1, c]
                    ) / 4
                    H3 = np.array([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
                    try:
                        offset = -np.linalg.solve(H3, grad)
                    except np.linalg.LinAlgError:
                        continue
                    if np.max(np.abs(offset)) > 1.5:
                        continue # offset ใหญ่เกินไป ไม่เสถียร -> ทิ้ง
                    D_hat = curr[r, c] + 0.5 * grad @ offset
                    if abs(D_hat) / img_range < contrast_thresh:
                        continue
                    H2 = np.array([[Dxx, Dxy], [Dxy, Dyy]])
                    tr, det = np.trace(H2), np.linalg.det(H2)
                    if det <= 0 or (tr**2 / det) >= edge_thresh:
                        continue
                    # ตำแหน่งจริงในภาพต้นฉบับ (คูณกลับด้วย downsample factor)
                    x = (c + offset[0]) * factor
                    y = (r + offset[1]) * factor
                    sigma_eff = sigmas[si] * (2 ** (offset[2] / len(sigmas))) * factor
                    keypoints.append(
                        {
                            "x": x,
                            "y": y,
                            "sigma": sigma_eff,
                            "octave": oi,
                            "r": r,
                            "c": c,
                            "si": si,
                            "response": abs(D_hat),
                        }
                    )
    return keypoints


def assign_orientations(keypoints, octaves, n_bins=36, peak_ratio=0.8):
    oriented = []
    for kp in keypoints:
        oc = octaves[kp["octave"]]
        L = oc["Ls"][kp["si"]]
        r, c = kp["r"], kp["c"]
        sigma_kp = oc["sigmas"][kp["si"]]
        sigma_w = 1.5 * sigma_kp
        radius = max(1, int(round(3 * sigma_w)))
        H, W = L.shape
        hist = np.zeros(n_bins)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if rr <= 0 or rr >= H - 1 or cc <= 0 or cc >= W - 1:
                    continue
                gx = L[rr, cc + 1] - L[rr, cc - 1]
                gy = L[rr + 1, cc] - L[rr - 1, cc]
                m = np.hypot(gx, gy)
                theta = np.degrees(np.arctan2(gy, gx)) % 360
                w = m * np.exp(-(dr**2 + dc**2) / (2 * sigma_w**2))
                b = int(theta // (360 / n_bins)) % n_bins
                hist[b] += w
        if hist.max() == 0:
            continue
        peak = hist.max()
        for i, v in enumerate(hist):
            if v >= peak_ratio * peak:
                theta_deg = i * (360 / n_bins) + (360 / n_bins) / 2
                kp2 = dict(kp)
                kp2["theta"] = theta_deg
                oriented.append(kp2)
    return oriented

def compute_descriptors(keypoints, octaves, n_blocks=4, n_bins=8, window_bins=4):

    descs = []
    valid_kps = []

    for kp in keypoints:

        oc = octaves[kp["octave"]]
        L = oc["Ls"][kp["si"]]

        r = kp["r"]
        c = kp["c"]

        theta_main = kp["theta"]
        sigma_kp = oc["sigmas"][kp["si"]]

        H, W = L.shape
        scale = sigma_kp

        # Descriptor window = 16 x scale
        half = int(np.ceil(8 * scale))

        if r - half < 1 or r + half >= H - 1 or c - half < 1 or c + half >= W - 1:
            continue

        histograms = np.zeros((n_blocks, n_blocks, n_bins), dtype=np.float64)

        theta_rad = np.deg2rad(theta_main)

        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        # Gaussian weight
        sigma_window = 0.5 * n_blocks * scale

        # ------------------------------------
        # วนพื้นที่รอบ Keypoint
        # ------------------------------------

        for dr in range(-half, half):

            for dc in range(-half, half):

                x_rot = cos_t * dc + sin_t * dr
                y_rot = -sin_t * dc + cos_t * dr

                # แปลงตำแหน่งไปยัง Block
                bin_x = x_rot / (window_bins * scale) + n_blocks / 2
                bin_y = y_rot / (window_bins * scale) + n_blocks / 2

                bi = int(np.floor(bin_y))
                bj = int(np.floor(bin_x))

                if not (0 <= bi < n_blocks and 0 <= bj < n_blocks):
                    continue

                rr = r + dr
                cc = c + dc

                if rr <= 0 or rr >= H - 1 or cc <= 0 or cc >= W - 1:
                    continue

                # ------------------------------------
                # Gradient
                # ------------------------------------

                gx = L[rr, cc + 1] - L[rr, cc - 1]
                gy = L[rr + 1, cc] - L[rr - 1, cc]

                magnitude = np.hypot(gx, gy)

                theta = np.degrees(np.arctan2(gy, gx)) % 360

                # Orientation relative to Keypoint
                theta_rel = (theta - theta_main) % 360

                # Orientation bin
                bin_ori = int(theta_rel / (360 / n_bins)) % n_bins

                # ------------------------------------
                # Gaussian Weight
                # ------------------------------------

                weight = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma_window**2))

                histograms[bi, bj, bin_ori] += magnitude * weight

        # ------------------------------------
        # Flatten
        # 4 x 4 x 8 = 128
        # ------------------------------------

        desc = histograms.flatten()
        norm = np.linalg.norm(desc)

        if norm < 1e-6:
            continue

        desc = desc / norm

        # Threshold
        desc = np.clip(desc, 0, 0.2)

        # Normalize อีกครั้ง
        norm = np.linalg.norm(desc)

        if norm < 1e-6:
            continue

        desc = desc / norm

        descs.append(desc)
        valid_kps.append(kp)

    return valid_kps, np.array(descs)

def detect_and_describe(img, n_octaves=3, sigma0=1, s=3):
    octaves = build_pyramid(img, n_octaves, sigma0, s)
    kps = find_extrema(octaves)
    kps = assign_orientations(kps, octaves)
    kps, descs = compute_descriptors(kps, octaves)
    return kps, descs

def match_descriptors(desc1, desc2, ratio_thresh=0.75):

    matches = []

    # ป้องกันกรณีไม่มี descriptor
    if len(desc1) == 0 or len(desc2) < 2:
        return matches

    for i in range(len(desc1)):

        # Euclidean Distance
        d = np.linalg.norm(desc1[i] - desc2, axis=1)

        order = np.argsort(d)

        n1 = order[0]
        n2 = order[1]

        d1 = d[n1]
        d2 = d[n2]

        ratio = d1 / (d2 + 1e-10)

        if ratio < ratio_thresh:
            matches.append((i, int(n1), float(d1), float(ratio)))

    return matches

def fit_affine(src, dst):
    A, B = [], []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0])
        B.append(xp)
        A.append([0, 0, 0, x, y, 1])
        B.append(yp)
    A, B = np.array(A), np.array(B)
    p, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, e, c, d, f = p
    return np.array([[a, b], [c, d]]), np.array([e, f])

def ransac_affine(src_pts, dst_pts, n_iter=2000, threshold=6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = len(src_pts)
    if n < 3:
        return None, None, []
    best_inliers = []
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        try:
            M, t = fit_affine(src_pts[idx], dst_pts[idx])
        except np.linalg.LinAlgError:
            continue
        pred = (M @ src_pts.T).T + t
        err = np.linalg.norm(pred - dst_pts, axis=1)
        inliers = np.where(err < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if len(best_inliers) < 3:
        return None, None, best_inliers
    M_final, t_final = fit_affine(src_pts[best_inliers], dst_pts[best_inliers])
    return M_final, t_final, best_inliers

def visualize(obj, scene, kps_obj, kps_scene, matches, inlier_idx, scene_box):

    # ====================================================
    # STEP 1: Object + Keypoints
    # ====================================================
    plt.figure(figsize=(8, 6))

    plt.imshow(obj, cmap="gray")

    for kp in kps_obj:

        r = max(2, kp["sigma"] * 2)

        circ = plt.Circle(
            (kp["x"], kp["y"]), r, color="lime", fill=False, linewidth=1.2
        )

        plt.gca().add_patch(circ)

        # วาด Orientation
        th = np.radians(kp["theta"])

        plt.plot(
            [kp["x"], kp["x"] + r * np.cos(th)],
            [kp["y"], kp["y"] + r * np.sin(th)],
            color="red",
            linewidth=1,
        )

        plt.title(f"STEP 1: Object + Keypoints ({len(kps_obj)} pts)")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("step1_object_keypoints.png", dpi=150, bbox_inches="tight")

    plt.show()

    print("บันทึก: step1_object_keypoints.png")

    # ====================================================
    # STEP 2: Scene + Keypoints
    # ====================================================
    plt.figure(figsize=(10, 7))

    plt.imshow(scene, cmap="gray")

    for kp in kps_scene:

        r = max(2, kp["sigma"] * 2)

        circ = plt.Circle((kp["x"], kp["y"]), r, color="lime", fill=False, linewidth=1)

        plt.gca().add_patch(circ)

    plt.title(f"STEP 2: Scene + Keypoints ({len(kps_scene)} pts)")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("step2_scene_keypoints.png", dpi=150, bbox_inches="tight")

    plt.show()

    print("บันทึก: step2_scene_keypoints.png")

    # ====================================================
    # STEP 3: Matches
    # ====================================================

    h1, w1 = obj.shape
    h2, w2 = scene.shape

    H = max(h1, h2)
    W = w1 + w2

    canvas = np.zeros((H, W))

    canvas[:h1, :w1] = obj
    canvas[:h2, w1 : w1 + w2] = scene

    plt.figure(figsize=(15, 8))

    plt.imshow(canvas, cmap="gray")

    inlier_set = set(inlier_idx.tolist()) if inlier_idx is not None else set()

    for mi, m in enumerate(matches):

        oi, si = m[0], m[1]

        x1 = kps_obj[oi]["x"]
        y1 = kps_obj[oi]["y"]

        x2 = kps_scene[si]["x"] + w1
        y2 = kps_scene[si]["y"]

        is_in = mi in inlier_set

        plt.plot(
            [x1, x2],
            [y1, y2],
            color="lime" if is_in else "red",
            alpha=0.9 if is_in else 0.3,
            linewidth=1,
        )

    n_in = len(inlier_idx) if inlier_idx is not None else 0
    plt.title(f"STEP 3: Matches " f"(green=inlier {n_in}, red=rejected)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("step3_matches.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("บันทึก: step3_matches.png")

    # ====================================================
    # STEP 4: Detected Object
    # ====================================================

    plt.figure(figsize=(10, 7))
    plt.imshow(scene, cmap="gray")
    poly = plt.Polygon(scene_box, closed=True, fill=False, edgecolor="red", linewidth=3)
    plt.gca().add_patch(poly)
    plt.title("STEP 4: Detected Object")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("step4_detected_object.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("บันทึก: step4_detected_object.png")