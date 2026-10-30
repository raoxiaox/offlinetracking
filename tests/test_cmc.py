import cv2
import numpy as np

from offlinetracking.cmc import CMCTrack, apply, estimate_pair


def test_estimate_pair_recovers_known_affine(cfg):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((360, 640)) * 255).astype(np.uint8), (5, 5), 0)
    ang, s, tx, ty = 0.01, 1.01, 4.0, -2.0
    M = np.array([[s * np.cos(ang), -s * np.sin(ang), tx], [s * np.sin(ang), s * np.cos(ang), ty]])
    img1 = cv2.warpAffine(img, M, (640, 360))
    H, rho, sigma = estimate_pair(img, img1, np.zeros((0, 4)), cfg["cmc"])
    p = np.array([[320.0, 180.0], [100.0, 300.0]])
    assert np.allclose(apply(H, p), p @ M[:, :2].T + M[:, 2], atol=0.5)
    assert rho > 0.3 and sigma < 1.0


def test_estimate_pair_too_few_points_fails(cfg):
    flat = np.full((100, 100), 128, np.uint8)
    H, rho, _ = estimate_pair(flat, flat, np.zeros((0, 4)), cfg["cmc"])
    assert rho == 0.0 and np.allclose(H, np.eye(3))


def test_cmctrack_composition_and_rho():
    n = 10
    H = np.tile(np.eye(3), (n, 1, 1))
    for t in range(1, n):
        H[t][0, 2] = 1.0  # 每帧向右平移 1 像素
    rho = np.ones(n)
    rho[5] = 0.2
    c = CMCTrack(H, rho)
    assert np.allclose(apply(c.Hab(2, 6), [0, 0]), [4, 0])
    assert np.allclose(apply(c.Hab(6, 2), [0, 0]), [-4, 0])
    assert c.rho(2, 5) == 1.0 and c.rho(2, 6) == 0.2
