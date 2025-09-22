use microflow::model;
use microflow::buffer::{Buffer4D, Buffer2D};
use nalgebra::SMatrix;
use image::{ImageBuffer, Luma};
use std::path::Path;

/// Int8 quantized TensorFlow Lite model for 48x64x3 RGB segmentation
#[model("models/Nano_U.tflite")]
struct NanoU;

/// Performs segmentation inference on a PNG image
///
/// # Parameters
/// * `image_path` - Path to the PNG image file
/// * `output_path` - Path to save the resulting segmentation mask
///
/// # Returns
/// * `Result<(), Box<dyn std::error::Error>>` - Ok if inference succeeds
fn predict(image_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Check if file exists
    if !Path::new(image_path).exists() {
        return Err(format!("Image not found: {}", image_path).into());
    }
    
    let img = image::open(image_path)?;
    let resized_img = image::imageops::resize(
        &img.to_rgb8(),
        64, 48,
        image::imageops::FilterType::Lanczos3
    );
    
    // Preprocess image with normalized quantization (matching Python training)
    let mut image_data: Vec<Vec<[i8; 3]>> = Vec::with_capacity(48);
    for y in 0..48 {
        let mut row: Vec<[i8; 3]> = Vec::with_capacity(64);
        for x in 0..64 {
            let pixel = resized_img.get_pixel(x, y);
            // Correct INT8 quantization: (normalized_value / scale) + zero_point
            // Scale: 0.003921569, Zero point: -128
            let rgb = [
                (((pixel[0] as f32 / 255.0) / 0.003921569) - 128.0).clamp(-128.0, 127.0) as i8,
                (((pixel[1] as f32 / 255.0) / 0.003921569) - 128.0).clamp(-128.0, 127.0) as i8,
                (((pixel[2] as f32 / 255.0) / 0.003921569) - 128.0).clamp(-128.0, 127.0) as i8,
            ];
            row.push(rgb);
        }
        image_data.push(row);
    }
    
    // Create Buffer4D for microflow
    let input_matrix: Buffer2D<[i8; 3], 48, 64> = SMatrix::from_fn(|r, c| image_data[r][c]);
    let input_buffer: Buffer4D<i8, 1, 48, 64, 3> = [input_matrix];

    let prediction = NanoU::predict_quantized(input_buffer);
    let output = &prediction[0];

    // Convert output to binary segmentation using threshold of 0.0
    let mut model_output_image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(
        output.ncols() as u32,
        output.nrows() as u32
    );
    
    let threshold = 0.0;
    for y in 0..output.nrows() {
        for x in 0..output.ncols() {
            let logit = output[(y, x)][0];
            let binary_value = if logit > threshold { 255u8 } else { 0u8 };
            model_output_image.put_pixel(x as u32, y as u32, Luma([binary_value]));
        }
    }
    
    let output_image = image::imageops::resize(
        &model_output_image,
        img.width(), img.height(),
        image::imageops::FilterType::Nearest
    );
    
    // Create the output directory if it doesn't exist.
    if let Some(parent_dir) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent_dir)?;
    }
    output_image.save(output_path)?;
    
    Ok(())
}

fn main() {
    use std::fs;
    let input_dir = "../Nano-U/data/processed_data/test/img";
    let output_dir = "output/try";

    // Collect all PNG files in the input directory
    let entries = match fs::read_dir(input_dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Failed to read input directory: {}", e);
            return;
        }
    };

    let mut image_files: Vec<_> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect();

    image_files.sort();
    let num_frames = image_files.len();
    println!("Starting inference for {} frames...", num_frames);

    for (i, input_path) in image_files.iter().enumerate() {
        let file_name = match input_path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => {
                eprintln!("  -> Skipping file with invalid name: {:?}", input_path);
                continue;
            }
        };
        let output_path = format!("{}/prediction_{}", output_dir, file_name);

        println!("[{}/{}] Processing {:?} -> {}...", i + 1, num_frames, input_path, output_path);

        match predict(input_path.to_str().unwrap(), &output_path) {
            Ok(()) => (),
            Err(e) => {
                eprintln!("  -> Error processing file {:?}: {}", input_path, e);
            }
        }
    }

    println!("\nInference run completed.");
}