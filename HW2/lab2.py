"""Lab 2 - skin detection.

a) skin masks (white=255 skin, black=0) for several methods on test images
b) evaluation on the Pratheepan dataset (confusion matrix + accuracy)
c) face localisation (minimal square) from the skin mask, no face detector

Usage:  python lab2.py            (runs everything, writes to ./output)
"""
import os
import glob
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "skin-20261101T154723Z-1-001", "skin")
DS = os.path.join(ROOT, "Face_Dataset")
IMG_DIRS = {
    "FacePhoto": os.path.join(DS, "Pratheepan_Dataset", "FacePhoto"),
    "FamilyPhoto": os.path.join(DS, "Pratheepan_Dataset", "FamilyPhoto"),
}
GT_DIRS = {
    "FacePhoto": os.path.join(DS, "Ground_Truth", "GroundT_FacePhoto"),
    "FamilyPhoto": os.path.join(DS, "Ground_Truth", "GroundT_FamilyPhoto"),
}
OUT = os.path.join(ROOT, "output")


# ---------------------------------------------------------------- methods
def skin_rgb(bgr):
    """Method 1: explicit RGB rules."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    m = ((r > 95) & (g > 40) & (b > 20) & (mx - mn > 15)
         & (np.abs(r - g) > 15) & (r > g) & (r > b))
    return m.astype(np.uint8) * 255


def skin_hsv(bgr):
    """Method 2: H in [0,50] deg, S in [0.23,0.68], V in [0.35,1]."""
    f = bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)  # H in [0,360], S,V in [0,1]
    h, s, v = cv2.split(hsv)
    m = (h >= 0) & (h <= 50) & (s >= 0.23) & (s <= 0.68) & (v >= 0.35) & (v <= 1)
    return m.astype(np.uint8) * 255


def to_ycbcr(bgr):
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    return y, cb, cr


def skin_ycbcr(bgr):
    """Method 3: Y > 80, Cb in [85,135], Cr in [135,180]."""
    y, cb, cr = to_ycbcr(bgr)
    m = (y > 80) & (cb > 85) & (cb < 135) & (cr > 135) & (cr < 180)
    return m.astype(np.uint8) * 255


def skin_paper(bgr):
    """Method 4: Kolkur et al. (reference 5), RGB-HSV-YCbCr combined rule.

    skin = (HSV rule AND RGB rule) OR (RGB rule AND YCbCr polygon rule).
    The paper's "A > 15" is the alpha channel (always opaque) so it is dropped.
    Y/Cb/Cr use the lab's conversion (the paper's eq. 4 has a typo).
    """
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    rgb = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r - g) > 15)
    f = bgr.astype(np.float32) / 255.0
    h, s, _ = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
    hsv = (h >= 0) & (h <= 50) & (s >= 0.23) & (s <= 0.68)
    y, cb, cr = to_ycbcr(bgr)
    ycc = ((cr > 135) & (cb > 85) & (y > 80)
           & (cr <= 1.5862 * cb + 20)
           & (cr >= 0.3448 * cb + 76.2069)
           & (cr >= -4.5652 * cb + 234.5652)
           & (cr <= -1.15 * cb + 301.75)
           & (cr <= -2.2857 * cb + 432.85))
    return ((rgb & hsv) | (rgb & ycc)).astype(np.uint8) * 255


class GaussianSkinModel:
    """Extra (not in the lab): statistical skin colour models, trained on the
    Pratheepan ground truth, for comparison against the rule-based methods:
      * 'bayes': histogram Bayes classifier P(skin|rgb) > 0.5 (32 bins/channel)
      * 'gauss': single Gaussian on (Cb,Cr) thresholded by Mahalanobis distance
    Train on a subset of images and evaluate on the rest for a fair estimate.
    """

    def __init__(self, bins=32):
        self.bins = bins
        self.hs = np.zeros((bins,) * 3)
        self.hn = np.zeros((bins,) * 3)

    def _idx(self, bgr):
        q = (bgr.reshape(-1, 3).astype(np.int32) * self.bins) // 256
        return q[:, 2], q[:, 1], q[:, 0]

    def fit(self, pairs):
        crcb = []
        for img, gt in pairs:
            r, g, b = self._idx(img)
            sk = gt.reshape(-1) > 127
            np.add.at(self.hs, (r[sk], g[sk], b[sk]), 1)
            np.add.at(self.hn, (r[~sk], g[~sk], b[~sk]), 1)
            _, cb, cr = to_ycbcr(img)
            crcb.append(np.stack([cb[gt > 127], cr[gt > 127]], 1))
        crcb = np.concatenate(crcb)
        self.mu = crcb.mean(0)
        self.icov = np.linalg.inv(np.cov(crcb.T))
        ns, nn = self.hs.sum(), self.hn.sum()
        self.ps = self.hs / max(ns, 1)   # P(rgb|skin)
        self.pn = self.hn / max(nn, 1)   # P(rgb|non-skin)
        self.prior = ns / (ns + nn)

    def bayes(self, bgr):
        r, g, b = self._idx(bgr)
        ps = self.ps[r, g, b] * self.prior
        pn = self.pn[r, g, b] * (1 - self.prior)
        m = ps > pn
        return (m.reshape(bgr.shape[:2]).astype(np.uint8)) * 255

    def gauss(self, bgr, thr=6.0):
        _, cb, cr = to_ycbcr(bgr)
        d = np.stack([cb, cr], -1) - self.mu
        md = np.einsum("...i,ij,...j->...", d, self.icov, d)
        return (md < thr).astype(np.uint8) * 255


# ------------------------------------------------------------- evaluation
def confusion(pred, gt):
    p, g = pred > 127, gt > 127
    tp = int(np.sum(p & g)); fn = int(np.sum(~p & g))
    fp = int(np.sum(p & ~g)); tn = int(np.sum(~p & ~g))
    return tp, fn, fp, tn


def print_cm(name, tp, fn, fp, tn):
    acc = (tp + tn) / max(tp + fn + fp + tn, 1)
    print(f"\n{name}")
    print(f"{'':22}{'Pred skin':>14}{'Pred non-skin':>16}")
    print(f"{'GT skin':22}{tp:>14}{fn:>16}")
    print(f"{'GT non-skin':22}{fp:>14}{tn:>16}")
    print(f"Accuracy = {acc:.4f}")
    return acc


def load_dataset():
    data = []
    for k, d in IMG_DIRS.items():
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            cands = glob.glob(os.path.join(GT_DIRS[k], stem + ".*"))
            if not cands:
                continue
            img = cv2.imread(p)
            gt = cv2.imread(cands[0], cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            if gt.shape != img.shape[:2]:
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            data.append((k, stem, img, gt))
    return data


# ------------------------------------------------------- face localisation
def find_face(bgr, mask):
    """Minimal square framing the face, from a skin mask.

    Clean the mask with morphology, take connected components, and pick the
    largest blob that is roughly face-shaped (prefer large, compact, upper).
    Returns (x, y, side) or None.
    """
    k = max(3, int(min(bgr.shape[:2]) * 0.02) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best, best_score = None, 0
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.005 * H * W:
            continue
        fill = a / float(w * h)
        aspect = h / float(w)
        # faces: aspect ~0.8-1.6, reasonably filled bounding box
        shape = np.exp(-((np.log(aspect / 1.2)) ** 2) / 0.3) * fill
        score = a * shape
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    x, y, w, h, _ = st[best]
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x0 = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
    y0 = int(np.clip(cy - side // 2, 0, max(H - side, 0)))
    return x0, y0, int(min(side, W, H))


# ------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    methods = {"rgb": skin_rgb, "hsv": skin_hsv, "ycbcr": skin_ycbcr,
               "paper": skin_paper}

    # train the statistical model on every other image, test on the rest
    data = load_dataset()
    print(f"Pratheepan images with ground truth: {len(data)}")
    train = [(i, g) for n, (_, _, i, g) in enumerate(data) if n % 2 == 0]
    test = [d for n, d in enumerate(data) if n % 2 == 1]
    model = GaussianSkinModel()
    model.fit(train)
    stat = {"stat_bayes": model.bayes, "stat_gauss": model.gauss}

    # a) masks for the test images
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
    for p in test_files:
        img = cv2.imread(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        panels = [img]
        for name, fn in {**methods, **stat}.items():
            m = fn(img)
            cv2.imwrite(os.path.join(OUT, f"{stem}_{name}.png"), m)
            panels.append(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(OUT, f"{stem}_all.png"), np.hstack(panels))
    print(f"Wrote masks for {len(test_files)} test images to {OUT}")

    # b) evaluation
    print("\n=== Evaluation on Pratheepan ===")
    print("(rule-based methods: all images; statistical: held-out images only)")
    for name, fn in methods.items():
        tot = np.zeros(4, dtype=np.int64)
        for _, _, img, gt in data:
            tot += confusion(fn(img), gt)
        print_cm(f"Method: {name}", *map(int, tot))
    for name, fn in stat.items():
        tot = np.zeros(4, dtype=np.int64)
        for _, _, img, gt in test:
            tot += confusion(fn(img), gt)
        print_cm(f"Method: {name} (held-out half)", *map(int, tot))
    print("\nFor a like-for-like comparison, rule-based on held-out half:")
    for name, fn in methods.items():
        tot = np.zeros(4, dtype=np.int64)
        for _, _, img, gt in test:
            tot += confusion(fn(img), gt)
        print_cm(f"Method: {name} (held-out half)", *map(int, tot))

    # c) face detection on portraits
    face_dir = os.path.join(OUT, "faces")
    os.makedirs(face_dir, exist_ok=True)
    found = 0
    faces = [d for d in data if d[0] == "FacePhoto"]
    for _, stem, img, _ in faces:
        mask = skin_ycbcr(img)
        r = find_face(img, mask)
        vis = img.copy()
        if r:
            x, y, s = r
            cv2.rectangle(vis, (x, y), (x + s, y + s), (0, 255, 0), 2)
            found += 1
        cv2.imwrite(os.path.join(face_dir, stem + ".png"), vis)
    print(f"\nFace localisation: square found in {found}/{len(faces)} portraits "
          f"(see {face_dir})")


if __name__ == "__main__":
    main()
