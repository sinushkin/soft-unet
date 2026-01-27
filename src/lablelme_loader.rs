use std::collections::HashMap;
use crate::types::{Contour, ContourCrop, ContourGroup, Image, RectSize, SuPoint, SuPoint2F};
use anyhow::{Result, bail, anyhow};
use base64::Engine;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;
use base64::engine::general_purpose::STANDARD;
use crate::configuration::OUTER_LABEL;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelMeLite {
    pub shapes: Vec<ShapeLite>,
    pub image_path: Option<String>,
    pub image_data: Option<String>,
    pub image_height: i32,
    pub image_width: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShapeLite {
    pub label: String,

    #[serde(deserialize_with = "de_points")]
    pub points: Vec<SuPoint2F>,
}

fn de_points<'de, D>(deserializer: D) -> Result<Vec<SuPoint2F>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Vec<[f64; 2]> = Vec::deserialize(deserializer)?;

    Ok(raw
        .into_iter()
        .map(|[x, y]| SuPoint2F {
            x: x as f32,
            y: y as f32,
        })
        .collect())
}

fn load_labelme_from_file(path: &PathBuf) -> Result<LabelMeLite> {
    let text = std::fs::read_to_string(path)?;
    let data: LabelMeLite = serde_json::from_str(&text)?;
    Ok(data)
}

pub fn load_labelme(json_path: &PathBuf) -> Result<Image> {
    let labelme = load_labelme_from_file(json_path)?;

    let original: Result<Mat> = {
        // ---- case base64 imageData ----
        if let Some(ref b64) = labelme.image_data {
            let bytes = STANDARD.decode(b64)?;

            let buf = Mat::from_slice(&bytes)?;

            let mat = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;

            if mat.empty() {
                bail!("Failed to decode base64 image");
            }
            Ok(mat)
        }
        // ---- case imagePath ----
        else if let Some(ref image_path) = labelme.image_path {
            if let Some(json_dir) = json_path.parent() {
                let full = json_dir.join(image_path.clone());

                let mat = imgcodecs::imread(full.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;

                if mat.empty() {
                    bail!("Failed to load image {:?}", full);
                }
                Ok(mat)
            }else{
                bail!("Failed to load image {:?}", image_path)
            }
        }else{
            bail!("Failed to load image {:?}", json_path)
        }
    };
    let original = original?;
    assert_eq!(original.cols(), labelme.image_width);
    assert_eq!(original.rows(), labelme.image_height);
    assert!(original.cols()>0);
    assert!(original.rows()>0);
    let size = RectSize::new(labelme.image_width as usize, labelme.image_height as usize);
    let contour_crops = match build_contours(size, labelme.shapes) {
        Ok(crops) => crops,
        Err(e) => {
            bail!("Failed to build contours for file {:?} {:?}", json_path, e)
        }
    };
    Ok(Image{
        original,
        size,
        contour_crops,
    })
}


fn build_contours(
    original_size: RectSize,
    shapes: Vec<ShapeLite>
) -> Result<Vec<ContourCrop>> {

    let mut map: HashMap<usize, (Option<Contour>, Vec<Contour>)> = HashMap::new();

    for shape in shapes {
        let (is_outer, prefix, idx) = parse_label(&shape.label)?;
        let entry = map.entry(idx).or_insert((None, Vec::new()));

        let contour = Contour {
            label: shape.label,
            label_prefix: prefix,
            points: shape.points,
        };

        if is_outer {
            if entry.0.is_some() {
                bail!("Duplicate outer contour for index {}", idx);
            }
            entry.0 = Some(contour);
        } else {
            entry.1.push(contour);
        }
    }

    let mut result = Vec::with_capacity(map.len());

    for (idx, (outer_opt, inners)) in map {

        let outer = outer_opt
            .ok_or_else(|| anyhow!("Missing outer contour for index {}", idx))?;

        // --- compute bounding box (based on outer)

        let (min_x, min_y, max_x, max_y) = contour_bbox(&outer.points);

        // clamp to image (optional safety)
        let min_x = min_x.max(0);
        let min_y = min_y.max(0);

        let max_x = max_x.min(original_size.width as i32 - 1);
        let max_y = max_y.min(original_size.height as i32 - 1);

        let width  = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;

        let offset = SuPoint {
            x: min_x as usize,
            y: min_y as usize,
        };

        // --- shift contours into crop coordinates

        let shifted_outer = shift_contour(outer, min_x, min_y);

        let shifted_inners = inners.into_iter()
            .map(|c| shift_contour(c, min_x, min_y))
            .collect();

        let group = ContourGroup {
            idx,
            outer: shifted_outer,
            inners: shifted_inners,
        };

        result.push(ContourCrop {
            offset,
            size: RectSize { width, height },
            contour_group: group
        });
    }

    Ok(result)
}



fn parse_label(label: &str) -> Result<(bool, String, usize)> {
    // split once at '_'. outer_1 -> (outer, 1)
    let mut it = label.splitn(2, '_');
    let label = it.next().ok_or_else(|| anyhow::anyhow!("Bad label"))?;
    let num  = it.next().ok_or_else(|| anyhow::anyhow!("Bad label"))?;

    Ok((OUTER_LABEL == label, label.to_string(), num.parse()?))
}

fn contour_bbox(points: &[SuPoint2F]) -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for p in points {
        let x = p.x as i32;
        let y = p.y as i32;

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

fn shift_contour(contour: Contour, dx: i32, dy: i32) -> Contour {
    Contour {
        label: contour.label,
        label_prefix: contour.label_prefix,
        points: contour.points.iter()
            .map(|p| SuPoint2F {
                x: p.x - dx as f32,
                y: p.y - dy as f32,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_test() -> Result<()> {
        let json_path = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let image = load_labelme(&json_path)?;
        assert_eq!(6, image.contour_crops.len());
        assert_eq!(725, image.size.width);
        assert_eq!(386, image.size.height);
        Ok(())
    }
}
