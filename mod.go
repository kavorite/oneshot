package induction

import (
	"gonum.org/v1/gonum/mat"
	"word2vec"
	"os"
)

func main() {
	istrm, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	word2vec.EmbedConf {
		Head: func(v, w int) {
			
		}
		Embed: func(t string, v []float32) {
			
		},
	}.EmbedBin(istrm)
}