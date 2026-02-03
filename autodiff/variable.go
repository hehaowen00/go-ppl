package autodiff

import "math"

// DualNumber represents a dual number for forward-mode AD
type DualNumber struct {
	Value float64 // Real part
	Deriv float64 // Differential part
}

// Variable wraps a dual number for user-friendly API
type Variable struct {
	dual DualNumber
}

// NewVariable creates a new variable with given value and derivative
func NewVariable(value, deriv float64) *Variable {
	return &Variable{dual: DualNumber{Value: value, Deriv: deriv}}
}

// NewScalar creates a scalar constant (derivative = 0)
func NewScalar(value float64) *Variable {
	return &Variable{dual: DualNumber{Value: value, Deriv: 0}}
}

// NewInput creates an input variable (derivative = 1)
func NewInput(value float64) *Variable {
	return &Variable{dual: DualNumber{Value: value, Deriv: 1}}
}

// Value returns the value of the variable
func (v *Variable) Value() float64 {
	return v.dual.Value
}

// Deriv returns the derivative of the variable
func (v *Variable) Deriv() float64 {
	return v.dual.Deriv
}

// Add performs addition v = v + v2
func (v *Variable) Add(v2 *Variable) {
	v.dual.Value += v2.dual.Value
	v.dual.Deriv += v2.dual.Deriv
}

// Sub performs subtraction v = v - v2
func (v *Variable) Sub(v2 *Variable) {
	v.dual.Value -= v2.dual.Value
	v.dual.Deriv -= v2.dual.Deriv
}

// Mul performs multiplication v = v * v2
func (v *Variable) Mul(v2 *Variable) {
	// Product rule: d(uv) = u'v + uv'
	v.dual.Deriv = v.dual.Deriv*v2.dual.Value + v.dual.Value*v2.dual.Deriv
	v.dual.Value *= v2.dual.Value
}

// Div performs division v = v / v2
func (v *Variable) Div(v2 *Variable) {
	// Quotient rule: d(u/v) = (u'v - uv')/v²
	if v2.dual.Value == 0 {
		panic("division by zero")
	}
	v.dual.Deriv = (v.dual.Deriv*v2.dual.Value - v.dual.Value*v2.dual.Deriv) / (v2.dual.Value * v2.dual.Value)
	v.dual.Value /= v2.dual.Value
}

// ----- Elementary Functions -----

// Sin computes sine of variable
func Sin(v *Variable) *Variable {
	return &Variable{dual: DualNumber{
		Value: sin(v.dual.Value),
		Deriv: cos(v.dual.Value) * v.dual.Deriv,
	}}
}

// Cos computes cosine of variable
func Cos(v *Variable) *Variable {
	return &Variable{dual: DualNumber{
		Value: cos(v.dual.Value),
		Deriv: -sin(v.dual.Value) * v.dual.Deriv,
	}}
}

// Exp computes exponential of variable
func Exp(v *Variable) *Variable {
	expVal := exp(v.dual.Value)
	return &Variable{dual: DualNumber{
		Value: expVal,
		Deriv: expVal * v.dual.Deriv,
	}}
}

// Log computes natural logarithm of variable
func Log(v *Variable) *Variable {
	if v.dual.Value <= 0 {
		panic("log of non-positive number")
	}
	return &Variable{dual: DualNumber{
		Value: log(v.dual.Value),
		Deriv: v.dual.Deriv / v.dual.Value,
	}}
}

// Pow computes v raised to the power of exponent (constant)
func Pow(v *Variable, exponent float64) *Variable {
	if v.dual.Value == 0 && exponent <= 0 {
		panic("invalid power operation")
	}
	powVal := pow(v.dual.Value, exponent-1)
	return &Variable{dual: DualNumber{
		Value: powVal * v.dual.Value,
		Deriv: exponent * powVal * v.dual.Deriv,
	}}
}

// ----- Helper Functions for Operations that Return New Variables -----

// Add returns a new variable that is the sum of v1 and v2
func Add(v1, v2 *Variable) *Variable {
	return &Variable{dual: DualNumber{
		Value: v1.dual.Value + v2.dual.Value,
		Deriv: v1.dual.Deriv + v2.dual.Deriv,
	}}
}

// Mul returns a new variable that is the product of v1 and v2
func Mul(v1, v2 *Variable) *Variable {
	return &Variable{dual: DualNumber{
		Value: v1.dual.Value * v2.dual.Value,
		Deriv: v1.dual.Deriv*v2.dual.Value + v1.dual.Value*v2.dual.Deriv,
	}}
}

// ----- Utility Functions -----

// Gradient computes the gradient of a function at a point
func Gradient(f func([]*Variable) *Variable, inputs []float64) []float64 {
	grad := make([]float64, len(inputs))

	for i := range inputs {
		// Create variables with derivative 1 for the i-th input
		vars := make([]*Variable, len(inputs))
		for j := range vars {
			if i == j {
				vars[j] = NewInput(inputs[j])
			} else {
				vars[j] = NewScalar(inputs[j])
			}
		}

		result := f(vars)
		grad[i] = result.Deriv()
	}

	return grad
}

// ----- Math helper functions -----

func sin(x float64) float64 {
	return math.Sin(x)
}

func cos(x float64) float64 {
	return math.Cos(x)
}

func exp(x float64) float64 {
	return math.Exp(x)
}

func log(x float64) float64 {
	return math.Log(x)
}

func pow(x, y float64) float64 {
	return math.Pow(x, y)
}
