// ============================================================
// RUST CONCEPT: `pub` — makes items visible outside this module.
//   Without `pub`, items are private to the file they're in.
//   Search: "rust visibility pub keyword"
// ============================================================

// ============================================================
// RUST CONCEPT: `struct` — defines a custom type with named fields.
//   Similar to a class with only data fields (no methods).
//   Methods are attached separately via `impl` blocks (see below).
//   Search: "rust struct definition"
// ============================================================

pub struct LogisticRegression {
    pub weights: Vec<f64>,
    // RUST CONCEPT: Vec<T> — a growable, heap-allocated array.
    //   Unlike arrays ([T; N]), Vec can grow and shrink.
    //   Search: "rust Vec vs array"
    pub bias: f64,
}

// ============================================================
// RUST CONCEPT: `impl` — attaches methods to a type.
//   Everything inside `impl TypeName { ... }` becomes a method on that type.
//   `Self` is an alias for the type name (here, LogisticRegression).
//   Search: "rust impl blocks methods"
// ============================================================

impl LogisticRegression {
    // ========================================================
    // CONSTRUCTOR — provided for you.
    //
    // RUST CONCEPT: `fn` — declares a function.
    //   Parameters have the form `name: Type`. Return type after `->`.
    //   Search: "rust fn syntax return type"
    //
    // RUST CONCEPT: `vec!` macro — shorthand to create a Vec.
    //   vec![0.0; n_features] creates a Vec of n_features zeros.
    //   Search: "rust vec macro repeat syntax"
    // ========================================================

    pub fn new(n_features: usize) -> Self {
        // RUST CONCEPT: `Self` — refers to the type being implemented.
        //   Here, Self means LogisticRegression.
        //   Search: "rust Self keyword impl"
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    // ========================================================
    // TASK 1: Implement the sigmoid function.
    //
    //   sigmoid(z) = 1 / (1 + e^(-z))
    //
    //   This squashes any real number into the range (0, 1).
    //   Used to interpret the raw score (logit) as a probability.
    //
    // HINT: In Rust, f64 values have methods. One of them is `.exp()`.
    //   Example: (-z_f64).exp() computes e^(-z).
    //   Search: "rust f64 methods exp"
    //
    //   Function signature to write:
    //     fn sigmoid(z: f64) -> f64
    //
    //   Note: no `pub` or `&self` needed — this is a standalone
    //   helper function inside the impl block.
    // ========================================================

    // ========================================================
    // TASK 2: Implement the dot product of two slices.
    //
    //   dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
    //
    // RUST CONCEPT: `&[f64]` — a slice. A borrowed view into a Vec
    //   or array. Slices are the idiomatic way to pass collections
    //   to functions — they don't take ownership.
    //   Search: "rust slices vs Vec borrowing"
    //
    // HINT: Use iterator methods chained together:
    //   - `.iter()` creates an iterator over a collection
    //   - `.zip()` pairs elements from two iterators
    //   - `.map()` transforms each element
    //   - `.sum()` adds everything up
    //
    //   Example skeleton:
    //     a.iter().zip(b.iter()).map(|(&ai, &bi)| ...).sum()
    //
    //   Search: "rust iterator zip map sum"
    //   Search: "rust closure syntax pipe"
    //
    //   Function signature to write:
    //     fn dot(a: &[f64], b: &[f64]) -> f64
    // ========================================================

    // ========================================================
    // TASK 3: Implement predict_proba — returns P(class=1 | x).
    //
    //   p = sigmoid( dot(x, weights) + bias )
    //
    // RUST CONCEPT: `&self` — immutable borrow of the struct.
    //   Lets you read fields (self.weights, self.bias) without
    //   taking ownership. The caller keeps their object.
    //   Search: "rust self &self methods borrowing"
    //
    //   Call the sigmoid() and dot() functions you wrote above.
    //   Access weights via `&self.weights`, bias via `self.bias`.
    //
    //   Function signature:
    //     pub fn predict_proba(&self, x: &[f64]) -> f64
    // ========================================================

    // ========================================================
    // TASK 4: Implement predict — returns 1 if probability >= 0.5, else 0.
    //
    //   Return type is u8 (unsigned 8-bit integer: 0 or 1).
    //
    // RUST CONCEPT: `if` is an *expression* in Rust, not a statement.
    //   It returns a value. No ternary operator needed.
    //
    //   Example:
    //     let value = if condition { 1 } else { 0 };
    //
    //   Search: "rust if expression returns value"
    //
    //   Function signature:
    //     pub fn predict(&self, x: &[f64]) -> u8
    // ========================================================

    // ========================================================
    // TASK 5: Complete the training loop (gradient descent).
    //
    //   The skeleton below handles the outer epoch loop and
    //   the inner sample loop. YOU fill in the gradient math.
    //
    //   For each sample (x_i, y_i) in each epoch:
    //     1. Compute prediction:   p = self.predict_proba(x_i)
    //     2. Compute error:        error = p - y_i (as f64)
    //     3. Update each weight:   self.weights[j] -= lr * error * x_i[j]
    //     4. Update bias:          self.bias -= lr * error
    //
    //   After processing all samples each epoch, compute and
    //   print the average binary cross-entropy loss:
    //
    //     L = -(1/N) * SUM[ y_i * ln(p_i) + (1-y_i) * ln(1-p_i) ]
    //
    //   Use f64 method `.ln()` for natural logarithm.
    //   Use `.max(1e-15)` to avoid log(0): p.ln().max(...) won't work;
    //   instead clamp: if p < epsilon, use epsilon. Search: "rust f64 clamp max"
    //
    // RUST CONCEPT: `&mut self` — mutable borrow of the struct.
    //   Required when the method modifies fields (updating weights/bias).
    //   Only ONE mutable borrow can exist at a time (Rust's safety rule).
    //   Search: "rust mutable borrowing &mut self ownership rules"
    //
    // RUST CONCEPT: `for` loop — iterates over anything that implements
    //   the Iterator trait. Ranges are written 0..n (exclusive) or
    //   0..=n (inclusive).
    //   Search: "rust for loop range iterator"
    //
    // RUST CONCEPT: `.enumerate()` — wraps an iterator to yield
    //   (index, value) pairs. Search: "rust iterator enumerate"
    //
    // RUST CONCEPT: `as f64` — type cast. Converts one numeric type
    //   to f64. Example: let x = 5_usize as f64;
    //   Search: "rust type casting as keyword numeric"
    //
    //   Function signature:
    //     pub fn train(&mut self, samples: &[Vec<f64>], labels: &[u8], lr: f64, epochs: usize)
    //
    //   Fill in the sections marked TODO below.
    // ========================================================

    pub fn train(&mut self, samples: &[Vec<f64>], labels: &[u8], lr: f64, epochs: usize) {
        let n = samples.len() as f64;

        for epoch in 0..epochs {
            // --- Inner loop: iterate over each sample ---
            for i in 0..samples.len() {
                let x_i = &samples[i];
                let y_i = labels[i];

                // TODO: Step 1 — compute prediction p using predict_proba
                // let p = ...

                // TODO: Step 2 — compute error (p - y_i as f64)
                // let error = ...

                // TODO: Step 3 — update each weight: self.weights[j] -= lr * error * x_i[j]
                // Hint: use a for loop over 0..self.weights.len()

                // TODO: Step 4 — update bias: self.bias -= lr * error
            }

            // --- Compute loss for this epoch ---
            // TODO: Compute binary cross-entropy loss over all samples.
            //   Accumulate the sum into a variable, then divide by n.
            //   Print every 50 epochs with:
            //     println!("Epoch {:>4} | Loss: {:.4}", epoch, loss);
            //
            // HINT: You'll need a mutable accumulator variable.
            //   let mut total_loss = 0.0;
            //   Then loop over samples again, compute p for each,
            //   add to total_loss using the BCE formula.
            //   Use .ln() for log, and protect against log(0)
            //   by clamping: p.max(1e-15).ln()
        }
    }
}
