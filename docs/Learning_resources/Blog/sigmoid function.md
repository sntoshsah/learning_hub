# Sigmoid Function Derivation

The **sigmoid function** is a common activation function used in machine learning and statistics, especially in binary classification and logistic regression.

---

## ✅ 1. Definition of Sigmoid Function

The sigmoid function is defined as:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

It maps any real-valued number into the range \( (0, 1) \), making it useful for probability-based tasks.

---

## ✏️ 2. Why This Form?

The sigmoid function arises naturally in logistic regression, where we model the **log-odds** of a binary outcome as a linear function of the input.

We want a function that:
- Outputs values between 0 and 1 (interpreted as probabilities),
- Is smooth and differentiable,
- Is monotonic (increasing),
- Approaches 1 as \( x \to \infty \), and 0 as \( x \to -\infty \).

---

## 💡 3. Derivation from Logistic Model

Start with the log-odds (logit) expression:

\[
\log\left(\frac{p}{1 - p}\right) = x
\]

Solve for \( p \):

\[
\frac{p}{1 - p} = e^x
\Rightarrow
p = e^x (1 - p)
\Rightarrow
p = e^x - e^x p
\Rightarrow
p(1 + e^x) = e^x
\Rightarrow
p = \frac{e^x}{1 + e^x}
\]

Rewriting:

\[
p = \frac{1}{1 + e^{-x}} = \sigma(x)
\]

---

## 📐 Derivative of the Sigmoid Function

Let:

\[
y = \sigma(x) = \frac{1}{1 + e^{-x}}
\]

We want to find:

\[
\frac{dy}{dx} = \frac{d}{dx} \left( \frac{1}{1 + e^{-x}} \right)
\]

---

### ✏️ Use the Chain Rule

It's easier to rewrite the expression as:

\[
y = (1 + e^{-x})^{-1}
\]

Differentiate using the chain rule:

\[
\frac{dy}{dx} = -1 \cdot (1 + e^{-x})^{-2} \cdot \frac{d}{dx}(1 + e^{-x})
\]

\[
\frac{dy}{dx} = - (1 + e^{-x})^{-2} \cdot (-e^{-x})
\]

\[
\frac{dy}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2}
\]

---

### 💡 Express in Terms of \( \sigma(x) \)

Recall:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad
1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}
\]

So:

\[
\frac{dy}{dx} = \sigma(x) \cdot (1 - \sigma(x))
\]

---

### ✅ Final Result

\[
\frac{d}{dx} \sigma(x) = \sigma(x) (1 - \sigma(x))
\]

---

## 🔁 Summary

| Property                 | Formula                                  |
|--------------------------|------------------------------------------|
| Sigmoid Function         | \( \sigma(x) = \frac{1}{1 + e^{-x}} \)     |
| Derivative               | \( \sigma'(x) = \sigma(x)(1 - \sigma(x)) \) |
| Range                    | \( (0, 1) \)                             |
| Applications             | Logistic Regression, Neural Networks     |

---

> 💡 *This function is widely used for binary classification tasks and as an activation function in shallow neural networks.*

