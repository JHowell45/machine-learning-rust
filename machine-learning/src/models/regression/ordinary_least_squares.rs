struct OrdinaryLeastSquares {
    intercept: f64,
    coef: Vec<f64>,
}

impl OrdinaryLeastSquares {
    pub fn new(intercept: f64, coef: Vec<f64>) -> Self {
        Self { intercept: intercept, coef: coef }
    }

    // pub fn fit(features: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {

    // }

    pub fn predict(&self, features: &Vec<f64>) -> f64 {
        return self.intercept + self.coef.iter().zip(features.iter()).map(|(coef, feature)| coef * feature ).sum::<f64>();
    }

    pub fn rss(&self, features: &Vec<Vec<f64>>, labels: &Vec<f64>) -> f64 {
        return labels.iter().zip(features.iter()).map(|(label, features)| (label - self.predict(features)).powi(2) ).sum::<f64>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sklearn_example_1() {
        let model = OrdinaryLeastSquares::new(0.0, vec![0.5, 0.5]);
        assert_eq!(0.0, model.predict(&vec![0.0, 0.0]));
    }

    #[test]
    fn test_sklearn_example_2() {
        let model = OrdinaryLeastSquares::new(0.0, vec![0.5, 0.5]);
        assert_eq!(1.0, model.predict(&vec![1.0, 1.0]));
    }

    #[test]
    fn test_sklearn_example_3() {
        let model = OrdinaryLeastSquares::new(0.0, vec![0.5, 0.5]);
        assert_eq!(2.0, model.predict(&vec![2.0, 2.0]));
    }

    #[test]
    fn test_rss() {
        let model = OrdinaryLeastSquares::new(0.0, vec![0.5, 0.5]);
        let features = vec![vec![0.0, 0.0]];
        let labels = vec![0.0, 1.0, 2.0];
        assert_eq!(0.0, model.rss(&features, &labels));
    }
}