// src/nepre.rs
use once_cell::sync::Lazy;
pub static NEPRE_F6: Lazy<[[f32; 20]; 20]> = Lazy::new(|| {
    let txt = include_str!("../data/nepre_f6_example.txt");
    let mut m = [[0.0; 20]; 20];
    for (i, line) in txt.lines().filter(|l| !l.starts_with('#')).enumerate() {
        for (j, val) in line.split_whitespace().enumerate() {
            m[i][j] = val.parse::<f32>().unwrap();
        }
    }
    m
});
pub fn pair(a: u8, b: u8) -> f32 {
    // a and b are already indices (0-19), not letters
    let ia = a as usize;
    let ib = b as usize;
    NEPRE_F6[ia][ib]
}
