use crate::types::{Contour, RawImage, RectSize, SuPoint2F};
use anyhow::{Result, bail};
use base64::Engine;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;
use base64::engine::general_purpose::STANDARD;


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
pub(crate) struct ShapeLite {
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

pub fn load_labelme(json_path: &PathBuf) -> Result<RawImage> {
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
    let contours: Vec<Contour> = labelme.shapes
        .into_iter()
        .map(|shape_lite| -> Result<Contour> {
            Ok(Contour {
                label: shape_lite.label,
                points: shape_lite.points,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RawImage{
        mat: original,
        size,
        contours,
    })
}




#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_test() -> Result<()> {
        let json_path = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let image = load_labelme(&json_path)?;
        assert_eq!(14, image.contours.len());
        assert_eq!(725, image.size.width);
        assert_eq!(386, image.size.height);
        Ok(())
    }
}
