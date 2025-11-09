use sprs::io::read_matrix_market;
use sprs::num_kinds::Pattern;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix: sprs::TriMat<Pattern> = read_matrix_market("../data/bcspwr01.mtx")?;
    println!("Matrix shape: {:?}", matrix.shape());
    println!("Number of non-zero entries: {}", matrix.nnz());
    Ok(())
}