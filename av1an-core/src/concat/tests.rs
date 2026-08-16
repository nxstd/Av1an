use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

use av_format::rational::Rational64;
use tempfile::tempdir;

use super::*;

fn ivf_header(frame_count: u32, width: u16, fourcc: [u8; 4]) -> [u8; 32] {
    let mut header = [0u8; 32];
    header[..4].copy_from_slice(b"DKIF");
    header[4..6].copy_from_slice(&0u16.to_le_bytes());
    header[6..8].copy_from_slice(&32u16.to_le_bytes());
    header[8..12].copy_from_slice(&fourcc);
    header[12..14].copy_from_slice(&width.to_le_bytes());
    header[14..16].copy_from_slice(&1080u16.to_le_bytes());
    header[16..20].copy_from_slice(&24u32.to_le_bytes());
    header[20..24].copy_from_slice(&1u32.to_le_bytes());
    header[24..28].copy_from_slice(&frame_count.to_le_bytes());
    header
}

fn write_ivf(
    path: &Path,
    declared_frames: u32,
    width: u16,
    fourcc: [u8; 4],
    frames: &[(u64, &[u8])],
) {
    let mut file = File::create(path).expect("chunk should be created");
    file.write_all(&ivf_header(declared_frames, width, fourcc))
        .expect("header should be written");
    for (timestamp, payload) in frames {
        file.write_all(&(payload.len() as u32).to_le_bytes())
            .expect("frame size should be written");
        file.write_all(&timestamp.to_le_bytes()).expect("timestamp should be written");
        file.write_all(payload).expect("payload should be written");
    }
}

fn output_frames(path: &Path) -> (u32, Vec<(u64, Vec<u8>)>) {
    let mut file = File::open(path).expect("output should exist");
    let mut header = [0u8; 32];
    file.read_exact(&mut header).expect("header should be readable");
    let count = u32::from_le_bytes(header[24..28].try_into().expect("slice length"));
    let mut frames = Vec::new();
    loop {
        let mut frame_header = [0u8; 12];
        match file.read(&mut frame_header[..1]).expect("frame header should be readable") {
            0 => break,
            _ => file
                .read_exact(&mut frame_header[1..])
                .expect("frame header should be complete"),
        }
        let size = u32::from_le_bytes(frame_header[..4].try_into().expect("slice length"));
        let timestamp = u64::from_le_bytes(frame_header[4..].try_into().expect("slice length"));
        let mut payload = vec![0; size as usize];
        file.read_exact(&mut payload).expect("payload should be complete");
        frames.push((timestamp, payload));
    }
    (count, frames)
}

#[test]
fn mkvmerge_options_json_no_audio() {
    let result = mkvmerge_options_json(
        &["00000.ivf".to_string(), "00001.ivf".to_string()],
        "output.mkv",
        None,
        Some(Rational64::new(30, 1)),
    )
    .expect("options call should succeed");
    assert_eq!(
        result,
        r#"["-o", "output.mkv", "--default-duration", "0:30/1fps", "[", "00000.ivf", "00001.ivf","]"]"#
    );
}

#[test]
fn mkvmerge_options_json_with_audio() {
    let result = mkvmerge_options_json(
        &["00000.ivf".to_string(), "00001.ivf".to_string()],
        "output.mkv",
        Some("audio.mkv"),
        Some(Rational64::new(30, 1)),
    )
    .expect("options call should succeed");
    assert_eq!(
        result,
        r#"["-o", "output.mkv", "audio.mkv", "--default-duration", "0:30/1fps", "[", "00000.ivf", "00001.ivf","]"]"#
    );
}

#[test]
fn two_valid_ivf_chunks_are_concatenated() {
    let temp = tempdir().expect("temp dir should be created");
    let encode = temp.path().join("encode");
    fs::create_dir(&encode).expect("encode dir should be created");
    write_ivf(&encode.join("00000.ivf"), 2, 1920, *b"AV01", &[
        (0, b"first"),
        (1, b"second"),
    ]);
    write_ivf(&encode.join("00001.ivf"), 2, 1920, *b"AV01", &[
        (0, b"third"),
        (1, b"fourth"),
    ]);

    let output = temp.path().join("output.ivf");
    ivf(&encode, &output).expect("chunks should concatenate");

    let (count, frames) = output_frames(&output);
    assert_eq!(count, 4);
    assert_eq!(frames, vec![
        (0, b"first".to_vec()),
        (1, b"second".to_vec()),
        (2, b"third".to_vec()),
        (3, b"fourth".to_vec()),
    ]);
}

#[test]
fn truncated_ivf_timestamp_returns_error_instead_of_panicking() {
    let temp = tempdir().expect("temp dir should be created");
    let encode = temp.path().join("encode");
    fs::create_dir(&encode).expect("encode dir should be created");
    let chunk = encode.join("00000.ivf");
    let mut file = File::create(&chunk).expect("chunk should be created");
    file.write_all(&ivf_header(1, 1920, *b"AV01"))
        .expect("header should be written");
    file.write_all(&1u32.to_le_bytes()).expect("size should be written");
    file.write_all(&[0; 6]).expect("partial timestamp should be written");

    let output = temp.path().join("output.ivf");
    let error = ivf(&encode, &output).expect_err("truncated timestamp should fail");
    assert!(error.to_string().contains("00000.ivf"));
    assert!(!output.exists());
}

#[test]
fn truncated_ivf_payload_returns_error() {
    let temp = tempdir().expect("temp dir should be created");
    let encode = temp.path().join("encode");
    fs::create_dir(&encode).expect("encode dir should be created");
    let chunk = encode.join("00000.ivf");
    let mut file = File::create(&chunk).expect("chunk should be created");
    file.write_all(&ivf_header(1, 1920, *b"AV01"))
        .expect("header should be written");
    file.write_all(&4u32.to_le_bytes()).expect("size should be written");
    file.write_all(&0u64.to_le_bytes()).expect("timestamp should be written");
    file.write_all(b"x").expect("partial payload should be written");

    let output = temp.path().join("output.ivf");
    assert!(ivf(&encode, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn incompatible_ivf_headers_return_error() {
    let temp = tempdir().expect("temp dir should be created");
    let encode = temp.path().join("encode");
    fs::create_dir(&encode).expect("encode dir should be created");
    write_ivf(&encode.join("00000.ivf"), 0, 1920, *b"AV01", &[]);
    write_ivf(&encode.join("00001.ivf"), 0, 1280, *b"AV01", &[]);

    let output = temp.path().join("output.ivf");
    let error = ivf(&encode, &output).expect_err("incompatible headers should fail");
    assert!(error.to_string().contains("00001.ivf"));
}

#[test]
fn empty_ivf_directory_returns_error() {
    let temp = tempdir().expect("temp dir should be created");

    assert!(ivf(temp.path(), &temp.path().join("output.ivf")).is_err());
}

#[test]
fn output_frame_count_uses_actual_frames() {
    let temp = tempdir().expect("temp dir should be created");
    let encode = temp.path().join("encode");
    fs::create_dir(&encode).expect("encode dir should be created");
    write_ivf(&encode.join("00000.ivf"), 99, 1920, *b"AV01", &[(
        0, b"actual",
    )]);

    let output = temp.path().join("output.ivf");
    ivf(&encode, &output).expect("chunk should concatenate");

    assert_eq!(output_frames(&output).0, 1);
}
