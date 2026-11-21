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

**Depth resolution.** Differentiating gives `ΔZ = Z² / (f·B) · Δd`. A one-pixel disparity error therefore causes a depth error that grows with the square of the distance, so stereo becomes poor at long range. Monocular box-height estimation, `ΔZ/Z = Δh/h`, has error that also grows with range, but only linearly in Z.

**Link to this project.** With one moving camera, two frames separated by known ego-motion form the same geometry, with the baseline equal to the distance moved. The same applies to tracking the change in a box's height over frames (DMODE, Agand et al. 2022). Either one is a natural next step after the optical-flow extra.
