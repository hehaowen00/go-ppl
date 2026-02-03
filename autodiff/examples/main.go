package main

import (
	"fmt"
	"go-ppl/autodiff"
)

func main() {
	// Create input variables
	x := autodiff.NewInput(2.0) // f(x) where x=2, dx/dx=1
	y := autodiff.NewInput(3.0) // f(y) where y=3, dy/dy=1

	// Compute z = x*y + sin(x)
	z1 := autodiff.Mul(x, y)  // x*y
	z2 := autodiff.Sin(x)     // sin(x)
	z := autodiff.Add(z1, z2) // x*y + sin(x)

	fmt.Printf("f(x,y)=x*y+sin(x) at x=2, y=3\n")
	fmt.Printf("Value: %f\n", z.Value())
	fmt.Printf("df/dx: %f\n", z.Deriv())

	// Reset for ∂f/∂y
	x = autodiff.NewScalar(2.0) // constant for y derivative
	y = autodiff.NewInput(3.0)  // input for y derivative

	z1 = autodiff.Mul(x, y)
	z2 = autodiff.Sin(x)
	z = autodiff.Add(z1, z2)

	fmt.Printf("df/dy: %f\n", z.Deriv())
}
