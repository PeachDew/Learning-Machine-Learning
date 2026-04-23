// ============================================================
// RUST CONCEPT: `mod` — declares a submodule.
//   `mod model;` tells Rust to look for `model.rs` in the same
//   directory (src/). Everything `pub` in model.rs is accessible
//   via `model::ItemName`.
//   Search: "rust modules mod keyword file structure"
// ============================================================

mod model;

// ============================================================
// RUST CONCEPT: `use` — brings items into scope so you don't
//   need to write the full path every time.
//   Without this, you'd write `model::LogisticRegression` everywhere.
//   Search: "rust use keyword import"
// ============================================================

use model::LogisticRegression;

use rand::Rng;

// ============================================================
// TASK 6: Implement the evaluation function.
//
//   Given a trained model, test features x_test, and test labels
//   y_test, compute and print:
//     - TP (true positive):  predicted=1, actual=1
//     - TN (true negative):  predicted=0, actual=0
//     - FP (false positive): predicted=1, actual=0
//     - FN (false negative): predicted=0, actual=1
//     - Accuracy  = (TP + TN) / total
//     - Precision = TP / (TP + FP)
//     - Recall    = TP / (TP + FN)
//
// RUST CONCEPT: Tuple — a fixed-size collection of heterogeneous
//   values. Written (a, b). Access via .0, .1, or destructuring.
//   Search: "rust tuples"
//
// HINT: Use `.iter().zip()` to iterate over two collections in
//   parallel. Example:
//     for (xi, &yi) in samples.iter().zip(labels.iter()) {
//         let pred = model.predict(xi);
//         // compare pred and yi, increment counters
//     }
//   Search: "rust iterator zip two collections"
//
// RUST CONCEPT: Mutable variables need `mut`.
//   let tp = 0;    // ERROR: cannot assign to immutable variable
//   let mut tp = 0; // OK
//   Search: "rust mut mutable variable"
//
// HINT: Print formatted output with the println! macro.
//   println!("Accuracy: {:.2}%", accuracy * 100.0);
//   The {} are placeholders. {:?} is debug format. {:.2} is 2 decimals.
//   Search: "rust println macro format specifiers"
//
//   Function signature:
//     fn evaluate(model: &LogisticRegression, samples: &[Vec<f64>], labels: &[u8])
// ========================================================

// ============================================================
// DONE: generate_data — creates synthetic 2D data.
//   Not the focus of the exercise, but read through it for
//   exposure to Rust patterns if interested.
// ============================================================

fn generate_data(n: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n {
        let x1: f64 = rng.gen_range(-1.0..1.0);
        let x2: f64 = rng.gen_range(-1.0..1.0);
        let label: u8 = if x1 + x2 > 0.0 { 1 } else { 0 };
        features.push(vec![x1, x2]);
        labels.push(label);
    }

    (features, labels)
}

fn main() {
    // ====================================================
    // RUST CONCEPT: Tuple destructuring.
    //   let (a, b) = some_function_returning_tuple();
    //   Searches: "rust tuple destructuring let"
    // ====================================================

    let (features, labels) = generate_data(200);

    // ====================================================
    // RUST CONCEPT: Slicing a Vec with &v[start..end].
    //   &features[..split]  — elements 0 to split-1
    //   &features[split..]  — elements split to end
    //   This borrows a portion of the Vec without copying.
    //   Search: "rust slice syntax range Vec"
    // ====================================================

    let split = 150;
    let x_train = &features[..split];
    let x_test = &features[split..];
    let y_train = &labels[..split];
    let y_test = &labels[split..];

    // ====================================================
    // RUST CONCEPT: `let mut` — mutable variable binding.
    //   Without `mut`, the variable cannot be reassigned or
    //   modified. Rust enforces this at compile time.
    //   Search: "rust let mut variable mutability"
    // ====================================================

    let mut model = LogisticRegression::new(2);

    // ====================================================
    // TASK 7: Call model.train() to train the model.
    //
    //   Try: learning rate = 0.1, epochs = 500
    //   The method signature is:
    //     train(&mut self, samples: &[Vec<f64>], labels: &[u8], lr: f64, epochs: usize)
    //
    //   Pass x_train and y_train as the data.
    // ====================================================

    // ====================================================
    // TASK 8: Call your evaluate() function.
    //
    //   Pass &model, x_test, y_test.
    // ====================================================

    // ====================================================
    // TASK 9: Print the learned weights and bias.
    //
    // HINT: println!("{:?}", &model.weights) prints the Vec
    //   in debug format (shows all elements).
    //
    // RUST CONCEPT: `#[derive(Debug)]` on a struct enables
    //   debug formatting with {:?}. You'll need to add this
    //   attribute to the struct definition in model.rs.
    //   Search: "rust derive Debug trait"
    //
    //   Also print model.bias separately.
    // ====================================================
}
