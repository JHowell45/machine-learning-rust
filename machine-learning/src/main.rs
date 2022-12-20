mod models;

use models::clustering::k_means::KMeans;

fn k_means() {
    let k_means = KMeans::new(3);
    println!("{:#?}", k_means);
}

fn main() {
    k_means();
}
