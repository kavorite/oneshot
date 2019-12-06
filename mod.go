package main

import (
    "math"
    "fmt"
    "flag"
    "strings"
	"gonum.org/v1/gonum/mat"
    "github.com/james-bowman/sparse"
    "io"
	"github.com/kavorite/induction/word2vec"
    "gopkg.in/jdkato/prose.v2"
	"os"
    "encoding/binary"
    "sort")

var (
    windowSize int
    corpusPath, embeddingPath, opath string
)

func cos(u, v mat.Vector) float64 {
    return mat.Dot(u, v) / mat.Norm(u, 2) / mat.Norm(v, 2)
}

type embeddings struct {
    *mat.Dense
    Tokens []string
    Vocab map[string]int
}

func (eb embeddings) embed(t string) mat.Vector {
    if i, ok := eb.Vocab[t]; ok {
        v := eb.RawRowView(i)
        return mat.NewVecDense(len(v), v)
    } else {
        return nil
    }
}

func readEmbeddings(path string) (rtn embeddings, err error) {
    istrm, err := os.Open(path)
    if err != nil {
        return
    }
    i := 0
    embedder := word2vec.Embedder {
		Head: func(wordc, dimen int) error {
			rtn.Dense = mat.NewDense(wordc, dimen, nil)
            rtn.Tokens = make([]string, wordc)
            rtn.Vocab = make(map[string]int, wordc)
            return nil
		},
		Embed: func(t string, v []float32) error {
            rtn.Tokens[i] = t
            rtn.Vocab[t] = i
            l := make([]float64, len(v))
            for j := range v {
                l[j] = float64(v[j])
            }
            rtn.SetRow(i, l)
			i++
            return nil
		},
	}
    if len(path) >= 4 && path[len(path)-4:] == ".bin" {
        err = embedder.EmbedBin(istrm)
    } else {
        err = embedder.EmbedText(istrm)
    }
    return
}

func (eb embeddings) placeholder() (rtn mat.Vector) {
    k, _ := eb.Dims()
    s := 1 / float64(k)
    v := mat.NewVecDense(k, make([]float64, k))
    rtn = v
    for t := range eb.Vocab {
        if u := eb.embed(t); u != nil {
            v.AddScaledVec(v, s, u)
        }
    }
    return
}

// linear regression
// http://cowlet.org/2016/08/23/linear-regression-in-rust.html
func regress(X mat.Matrix, Y mat.Matrix) *mat.Dense {
    svd := new(mat.SVD)
    svd.Factorize(X, mat.SVDThin)
    U := new(mat.Dense)
    V := new(mat.Dense)
    A := new(mat.Dense)
    svd.UTo(U)
    svd.VTo(V)
    A.Mul(U.T(), Y)
    _, orderp1 := X.Dims()
    s := svd.Values(nil)[:orderp1]
    raw := make([]float64, orderp1)
    for i := 0; i < orderp1; i++ {
        raw[i] = A.At(i, 0) / s[i]
    }
    sInvA := mat.NewDense(orderp1, 1, raw)
    V.Mul(V, sInvA)
    return V
}

type Window []prose.Token

func (D Window) NGrams(n int, forEach func(Window)) {
    k := len(D) - n + 1
    for i := 0; i < k; i++ {
        forEach(D[i:i+n])
    }
}

func coocs(corpus io.Reader, eb embeddings, n int) (J mat.Matrix, F mat.Vector, err error) {
    v, _ := eb.Dims()
    juxt := sparse.NewDOK(v, v)
    freqs := make([]float64, v)
    err = readCorpus(corpus, func(src string) error {
        D, err := prose.NewDocument(src)
        if err != nil {
            return err
        }
        Window(D.Tokens()).NGrams(n, func(ctx Window) {
            incr := 1 / float64(n)
            for _, t := range ctx {
                if i, ok := eb.Vocab[t.Text]; ok {
                    freqs[i] += incr
                    for _, w := range ctx {
                        if j, ok := eb.Vocab[w.Text]; ok {
                            juxt.Set(i, j, juxt.At(i, j) + incr)
                        }
                    }
                }
            }
        })
        return nil
    })
    if err == nil {
        J = juxt
        F = mat.NewVecDense(v, freqs)
    }
    return
}

func readCorpus(istrm io.Reader, forEach func(string) error) error {
    var (
        doc string
        err error
    )
    for ; err != nil; _, err = fmt.Fscanf(istrm, "%s\n\n", &doc) {
        if err = forEach(doc); err != nil {
            return err
        }
    }
    if err == io.EOF {
        err = nil
    }
    return err
}

// https://github.com/NLPrinceton/ALaCarte/blob/master/compute.py
// wordCoocs: V × V
// embeddings: V × d
// wordFreqs: 1 × V
// weights: 1 × V (optional)
func induce(wordCoocs mat.Matrix, eb mat.Matrix, wordFreqs mat.Vector) *mat.Dense {
    v, d := eb.Dims()
    A := mat.NewDense(v, d, make([]float64, v*d))
    A.Mul(wordCoocs, eb)
    A.DivElem(A, wordFreqs)
    return regress(A, eb)
}

func persist(ostrm io.Writer, A *mat.Dense) (err error) {
    m, n := A.Dims()
    if _, err = fmt.Fprintf(ostrm, "%dx%d\n", m, n); err != nil {
        return
    }
    for i := 0; i < m-1; i++ {
        if err = binary.Write(ostrm, binary.LittleEndian, A.RawRowView(i)); err != nil {
            return
        }
    }
    return
}

func main() {
    flag.StringVar(&corpusPath, "corpus", "", "path to corpus")
    flag.StringVar(&embeddingPath, "embeddings", "", "path to embeddings")
    flag.StringVar(&opath, "opath", "./A.bin", "path to persist induction matrix")
    flag.IntVar(&windowSize, "n", 5, "n-gram context size")
    flag.Parse()
    if corpusPath == "" {
        fmt.Fprintf(os.Stderr, "please provide -corpus\n")
        os.Exit(1)
    }
    if embeddingPath == "" {
        fmt.Fprintf(os.Stderr, "please provide -embeddings\n")
        os.Exit(2)
    }
    eb, err := readEmbeddings(embeddingPath)
    if err != nil {
        panic(err)
    }
    istrm, err := os.Open(corpusPath)
    if err != nil {
        panic(err)
    }
    J, F, err := coocs(istrm, eb, windowSize)
    if err != nil {
        panic(err)
    }
    ostrm, err := os.OpenFile(opath, os.O_CREATE|os.O_WRONLY, 0666)
    if err != nil {
        panic(err)
    }
    A := induce(J, eb, F)
    if err = persist(ostrm, A); err != nil {
        panic(err)
    }
    if _, err = istrm.Seek(0, io.SeekStart); err != nil {
        panic(err)
    }
    placeholder := eb.placeholder()
    samples := make([]float64, 0, 1024)
    err = readCorpus(istrm, func(src string) error {
        doc, err := prose.NewDocument(strings.ToLower(src))
        if err != nil {
            return err
        }
        Window(doc.Tokens()).NGrams(windowSize, func(ctx Window) {
            v := eb.embed(ctx[len(ctx) / 2].Text)
            if v == nil {
                return
            }
            _, d := eb.Dims()
            vAvg := mat.NewVecDense(d, make([]float64, d))
            s := 1 / float64(len(ctx))
            for _, t := range ctx {
                v := eb.embed(t.Text)
                if v == nil {
                    v = placeholder
                }
                vAvg.AddScaledVec(vAvg, s, v)
            }
            vHat := mat.NewDense(1, d, make([]float64, d))
            vHat.Mul(A, vAvg.T())
            samples = append(samples, cos(vHat.RowView(0), v))
        })
        return nil
    })
    if err != nil {
        panic(err)
    }
    sort.Slice(samples, func(i, j int) bool {
        return samples[i] < samples[j]
    })
    median := samples[len(samples)/2]
    var sigma float64
    for _, x := range samples {
        r := median - x
        sigma += r*r
    }
    sigma = math.Sqrt(sigma)
    fmt.Printf("μ = %f; σ = %f\n", median, sigma)
}
