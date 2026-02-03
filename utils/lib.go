package utils

import "math/rand/v2"

func DefaultPCG() *rand.Rand {
	return rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
}
