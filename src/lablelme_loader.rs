use std::collections::HashMap;
use crate::types::{Contour, ContoursStruct, Image, Point};
use anyhow::{Result, bail};
use base64::Engine;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;
use base64::engine::general_purpose::STANDARD;

const OUTER_LABEL: &str = "outer";

#[derive(Debug, Serialize, Deserialize)]
struct LabelMeLite {
    pub shapes: Vec<ShapeLite>,
    pub imagePath: Option<String>,
    pub imageData: Option<String>,
    pub imageHeight: i32,
    pub imageWidth: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShapeLite {
    pub label: String,

    #[serde(deserialize_with = "de_points")]
    pub points: Vec<Point>,
}

fn de_points<'de, D>(deserializer: D) -> Result<Vec<Point>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Vec<[f64; 2]> = Vec::deserialize(deserializer)?;

    Ok(raw
        .into_iter()
        .map(|[x, y]| Point {
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
        if let Some(ref b64) = labelme.imageData {
            let bytes = STANDARD.decode(b64)?;

            let buf = Mat::from_slice(&bytes)?;

            let mat = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;

            if mat.empty() {
                bail!("Failed to decode base64 image");
            }
            Ok(mat)
        }
        // ---- case imagePath ----
        else if let Some(ref image_path) = labelme.imagePath {
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
    assert_eq!(original.cols(), labelme.imageWidth);
    assert_eq!(original.rows(), labelme.imageHeight);
    assert!(original.cols()>0);
    assert!(original.rows()>0);
    Ok(Image{
        original,
        width: labelme.imageWidth as usize,
        height: labelme.imageHeight as usize,
        contours: build_contours(labelme.shapes)?,
    })
}


fn build_contours(shapes: Vec<ShapeLite>) -> Result<Vec<ContoursStruct>> {
    // group by index postfix
    let mut map: HashMap<i32, (Option<Contour>, Vec<Contour>)> = HashMap::new();

    for shape in shapes {
        let (outer, idx) = parse_label(&shape.label)?;
        let entry = map.entry(idx).or_insert((None, Vec::new()));
        match outer {
            true => {
                if entry.0.is_some() {
                    bail!("Duplicate outer contour for index {}", idx);
                }
                entry.0 = Some(Contour{ label: shape.label, points: shape.points});
            }
            false => {
                entry.1.push(Contour{ label: shape.label, points: shape.points});
            }
        }
    }

    // build final vector
    let mut result = Vec::with_capacity(map.len());

    for (idx, (outer_opt, inners)) in map {
        let outer = outer_opt
            .ok_or_else(|| anyhow::anyhow!("Missing outer contour for index {}", idx))?;
        result.push(ContoursStruct {
            outer,
            inners,
        });
    }

    Ok(result)
}



fn parse_label(label: &str) -> Result<(bool, i32)> {
    // split once at '_'
    let mut it = label.splitn(2, '_');

    let kind = it.next().ok_or_else(|| anyhow::anyhow!("Bad label"))?;
    let num  = it.next().ok_or_else(|| anyhow::anyhow!("Bad label"))?;

    let idx: i32 = num.parse()?;
    let outer = match kind {
        OUTER_LABEL => true,
        _ => false
    };

    Ok((outer, idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_test() -> Result<()> {
        let json_path = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let image = load_labelme(&json_path)?;
        assert_eq!(6, image.contours.len());
        assert_eq!(725, image.width);
        assert_eq!(386, image.height);
        Ok(())
    }
}
