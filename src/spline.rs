use crate::types::SuPoint;

pub fn smooth_polygon(points: &[SuPoint], smooth_factor: usize) -> Vec<SuPoint> {
    if points.len() < 4 {
        return points.to_vec();
    }

    // ---- close contour if not closed ----
    let mut pts = points.to_vec();
    if pts[0].x != pts[pts.len() - 1].x || pts[0].y != pts[pts.len() - 1].y {
        pts.push(pts[0]);
    }

    let n = pts.len();

    let mut out = Vec::with_capacity(smooth_factor * n);

    for i in 0..n - 1 {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];

        for j in 0..smooth_factor {
            let t = j as f32 / smooth_factor as f32;

            let x = catmull_rom(p0.x as f32, p1.x as f32, p2.x as f32, p3.x as f32, t);
            let y = catmull_rom(p0.y as f32, p1.y as f32, p2.y as f32, p3.y as f32, t);

            out.push(SuPoint {
                x,
                y,
            });
        }
    }

    out
}

#[inline]
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    0.5 * (
        2.0 * p1 +
            (-p0 + p2) * t +
            (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2 +
            (-p0 + 3.0*p1 - 3.0*p2 + p3) * t3
    )
}


#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use std::path::Path;
    use opencv::{imgcodecs, imgproc};
    use opencv::prelude::{Mat, MatExprTraitConst};
    use anyhow::Result;
    use opencv::core::{Point, Scalar};
    use opencv::core::Vector;
    use super::*;
    #[test]
    fn test_smooth_polygon_hexagon() -> Result<()>{
        // центр и сторона
        let cx = 250.0;
        let cy = 250.0;
        let side = 100.0;

        // генерируем вершины шестиугольника
        let hexagon: Vec<SuPoint> = (0..6)
            .map(|i| {
                let angle = std::f32::consts::PI / 3.0 * i as f32;
                SuPoint::new(
                    (cx/2f32 + side * angle.cos()) as i32,
                    (cy/2f32 + side * angle.sin()) as i32,
                )
            })
            .collect();

        let smooth_factor = 10;

        let smooth = smooth_polygon(&hexagon, smooth_factor);

        // 1. Output has more points than input
        assert!(smooth.len() > hexagon.len(), "Smoothing should increase points");

        // 2. Output points are all Point type (trivial)
        for p in &smooth {
            assert!(p.x.is_finite());
            assert!(p.y.is_finite());
        }

        // 3. First and last point are roughly equal (closed polygon)
        let first = smooth.first().unwrap();
        let last  = smooth.last().unwrap();

        let dx = (first.x - last.x).abs();
        let dy = (first.y - last.y).abs();
        assert!(dx <= 10f32 && dy <= 10f32, "Polygon should be closed (first ≈ last)");

        println!("{:?}", smooth);

        // 3️⃣ Create canvas
        let mut canvas = Mat::zeros(cx as i32, cy as i32, opencv::core::CV_8UC3)?.to_mat()?;

        // 4️⃣ Convert hexagon to OpenCV Point
        let hex_pts: Vec<Point> = hexagon.into_iter().map(|p| p.into()).collect::<Vec<_>>();
        let smooth_pts: Vec<Point> = smooth.into_iter().map(|p| p.into()).collect::<Vec<_>>();

        imgproc::polylines(
            &mut canvas,
            &Vector::from_slice(&hex_pts.as_slice()),
            true,                      // closed
            Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue
            2,                         // thickness
            imgproc::LINE_AA,
            0,
        )?;

        // 6️⃣ Draw smoothed polygon in red
        imgproc::polylines(
            &mut canvas,
            &Vector::from_slice(&smooth_pts.as_slice()),
            true,
            Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
            2,
            imgproc::LINE_AA,
            0,
        )?;

        let out_dir = Path::new("out");
        create_dir_all(out_dir)?;
        // 7️⃣ Save
        let out_path = out_dir.join("smooth.png");
        imgcodecs::imwrite(out_path.to_str().unwrap(), &canvas, &Vector::new())?;

        Ok(())
    }
}
