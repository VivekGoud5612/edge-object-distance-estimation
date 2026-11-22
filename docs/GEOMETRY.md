# Geometry notes

This file has the maths behind `src/distance.py` and `src/bev.py`, plus the stereo derivation the brief lists as extra credit. Each formula names the code that uses it.

## 1. Pinhole camera model

A camera with focal length `f` (pixels) and principal point `(cx, cy)` maps a 3D point `(X, Y, Z)` in camera coordinates (x right, y down, z forward) to the pixel

```
u = fx · X / Z + cx
v = fy · Y / Z + cy
```

**Focal length in pixels.** `f_px = f_mm · image_width_px / sensor_width_mm`. Without a datasheet, use the horizontal field of view: `f_px = (W/2) / tan(HFOV/2)`. BDD100K has W = 1280. With an assumed HFOV of 65° that gives f ≈ 1005 px (`configs/camera.yaml`).

**Resolution.** `f_px` scales with the image. A camera calibrated at 1920 px wide and used at 1280 px has 2/3 of the focal length in pixels. `CameraIntrinsics.scaled_to()` makes this adjustment.

## 2. Distance from known object height (main method)

An object of real height `H`, standing at depth `Z`, has its top and bottom at `Y1` and `Y2 = Y1 + H`. Its image height is

```
h = v2 − v1 = fy · (Y2 − Y1) / Z = fy · H / Z      ⇒      Z = fy · H / h
```

These are similar triangles: the object and its image subtend the same angle at the camera centre.

- The height is used rather than the width, because the width of a barrier or a sign changes with viewing angle and its height doesn't.
- The formula assumes the object is roughly parallel to the image plane, which holds for objects ahead of the robot.
- Code: `pinhole_distance()`.

## 3. Distance from the ground plane (cross-check)

The camera sits at height `h_cam` above flat ground and is level. A ground point at depth `Z` has `Y = h_cam`, so

```
v = fy · h_cam / Z + cy      ⇒      Z = fy · h_cam / (v − cy)
```

Here `v` is the row of the object's bottom edge, where it touches the road. `cy` is the horizon row for a level camera.

- If the camera is pitched down by θ, the horizon rises to `y0 = cy − fy · tan θ`.
- This method needs no object size, so its errors are independent of the pinhole method's. When the two agree, both are probably right.
- It does not apply to stop signs, which sit on poles.
- Code: `ground_plane_distance()`, `horizon_row()`.

## 4. Error propagation

The distance is `Z = f · H / h`, a product and a quotient. For a product or quotient, relative errors add in quadrature:

```
ΔZ/Z = sqrt( (ΔH/H)² + (Δf/f)² + (Δh/h)² )
```

- An error in `f` or `H` shifts every distance by the same percentage. A 10% error in the assumed height gives a 10% error at any range.
- A fixed box error of about 2 px is a larger fraction of `h` the farther away the object is. A cone at 40 m is about 18 px tall, so 2 px is already about 11%.
- Code: `relative_uncertainty()`. For a table of these numbers, run `python -m scripts.validate_distance --sensitivity`.

## 5. Bird's-eye view: homography

All points on a plane map to the image through a 3×3 homography `H`, defined up to scale:

```
[u v 1]ᵀ ~ H · [X Z 1]ᵀ
```

Four point pairs, with no three in a line, determine `H` (8 unknowns, 2 equations per pair).

1. Project 4 ground corners (in metres) into the image using the camera model in §1. A pitched camera is handled by rotating about the x-axis: `y_c = y cos θ − z sin θ`, `z_c = y sin θ + z cos θ`.
2. Pass the 4 pairs to `cv2.getPerspectiveTransform`.
3. Warp the frame with `cv2.warpPerspective`. The road then appears top-down, at a fixed number of pixels per metre.

Code: `src/bev.py`. Unit test: `tests/test_bev.py` checks that this projection inverts the formula in §3.

## 6. Extra credit: epipolar geometry and the disparity–depth relation

Take two identical cameras with focal length `f`, mounted side by side a baseline `B` apart, rectified so that their image rows line up.

1. A point `P = (X, Y, Z)` in the left camera's frame is `(X − B, Y, Z)` in the right camera's frame.
2. The projections are `u_L = f·X/Z + cx` and `u_R = f·(X − B)/Z + cx`.
3. The **disparity** is `d = u_L − u_R = f·B / Z`, which gives

```
Z = f · B / d
```

**Epipolar constraint.** For any two views, corresponding points satisfy `x_Rᵀ F x_L = 0`, where `F` is the fundamental matrix, or `E = [t]ₓ R` in normalised coordinates. For a rectified pair, `R = I` and `t = (B, 0, 0)`. The constraint then reduces to `v_L = v_R`: the match for a pixel lies on the same row. That is why stereo matching is a 1D search along rows.

**Depth resolution.** Differentiating `Z = f·B/d` gives `|ΔZ| = Z² / (f·B) · |Δd|`. A one-pixel disparity error therefore causes a depth error that grows with the *square* of the distance.

**The monocular method has the same form.** From `Z = f·H/h`, a pixel error in box height gives `|ΔZ| = Z² / (f·H) · |Δh|`. The known object height `H` plays the role of the baseline `B`. A 0.70 m cone is in effect a 0.70 m "baseline", which is larger than most stereo rigs. The catch is that `H` itself is uncertain. The analysis in FINDINGS.md finds real height spread of ×/÷ 1.45 per box, and that systematic error dwarfs the pixel term. Stereo has no such term: `B` is known to a millimetre after calibration.

Error for 1 px of disparity or box-height error, with f = 1005 px. These are the numbers from `src/stereo.py`:

| Z (m) | stereo B = 0.12 m | stereo B = 0.30 m | mono, 0.70 m cone (pixel term only) |
|---|---|---|---|
| 2 | 0.03 m | 0.01 m | 0.01 m |
| 5 | 0.21 m | 0.08 m | 0.04 m |
| 10 | 0.83 m | 0.33 m | 0.14 m |
| 20 | 3.3 m | 1.3 m | 0.57 m |
| 40 | 13.3 m | 5.3 m | 2.3 m |

At 10 m the height spread alone adds roughly ±4 m to the mono estimate. A calibrated stereo pair's 1 px error only catches up with mono's height error (about 0.4·Z) where `Z²/(f·B) = 0.4·Z`, which is about 48 m for B = 0.12 m and 120 m for B = 0.30 m. Inside the detector's useful range (about 20 m), stereo is clearly more accurate.

**Motion stereo with one camera.** Suppose the robot drives forward a known distance `B` between two frames (from odometry), and the object is static. Then `Z₁ = Z₂ + B`, with `h₁ = f·H/Z₁` and `h₂ = f·H/Z₂`, so

```
h₂ / h₁ = Z₁ / Z₂ = (Z₂ + B) / Z₂   ⇒   Z₂ = B · h₁ / (h₂ − h₁)
```

Neither `f` nor `H` appears. The driven distance is the baseline and the box growth `h₂ − h₁` is the "disparity", with the same `1/disparity` error behaviour. It needs the same object in both frames, which is what the optical-flow tracker provides (`src/tracking.py`, `distance_from_scale_change`). It is only usable once the box has grown by several pixels. It also removes the dominant error source of the single-frame method, the unknown real height. Related work: DMODE, Agand et al. 2022.

**Epipolar view of the same thing.** For pure forward motion, `R = I` and `t = (0, 0, B)`. The epipole is the focus of expansion at the principal point, and epipolar lines are rays from it. A static point slides outward along its ray, and the distance it slides gives depth, just as disparity does along a row in the rectified case. `src/stereo.py` has `essential_matrix(R, t)`, and `tests/test_stereo.py` checks numerically that `x_Rᵀ E x_L = 0`.
