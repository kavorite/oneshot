package main

import (
    "encoding/binary"
    "flag"
    "fmt"
    "unicode"
    "io"
    "math"
    "os"
    "sort"
    "strings"
    "bufio"
    "sync"
    "golang.org/x/text/transform"
    "golang.org/x/text/unicode/norm"

    "github.com/james-bowman/sparse"
    "github.com/kavorite/induction/word2vec"
    "gonum.org/v1/gonum/mat"
)

var (
    windowSize                       int
    corpusPath, embeddingPath, opath string
)

func cos(u, v mat.Vector) float64 {
    return mat.Dot(u, v) / mat.Norm(u, 2) / mat.Norm(v, 2)
}

func pbar(task string) func(float64) {
    const width = 24
    return func(x float64) {
        var bar strings.Builder
        bar.Grow(width + 2)
        bar.WriteByte('[')
        n := int(x * width)
        for i := 1; i <= n; i++ {
            bar.WriteByte('#')
        }
        for i := 1; i <= width-n; i++ {
            bar.WriteByte(' ')
        }
        bar.WriteByte(']')
        fmt.Printf("%s: %s %.2f%%\r", task, bar.String(), x*100)
    }
}

type embeddings struct {
    *mat.Dense
    Tokens []string
    Vocab  map[string]int
}

func (eb embeddings) embed(t string) (v mat.Vector) {
    if i, ok := eb.Vocab[t]; ok {
        v = eb.RowView(i)
    }
    return
}

func loadEmbeddings(istrm io.ReadSeeker, binary bool, progress func(float64)) (rtn embeddings, err error) {
    strmLength, err := istrm.Seek(0, io.SeekEnd)
    if err != nil {
        return
    }
    if _, err = istrm.Seek(0, io.SeekStart); err != nil {
        return
    }
    bytesRead := int64(0)
    i := 0
    embedder := word2vec.Embedder{
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
            if progress != nil {
                if bytesRead, err = istrm.Seek(0, io.SeekCurrent); err != nil {
                    return err
                }
                if progress != nil {
                    progress(float64(bytesRead) / float64(strmLength))
                }
            }
            return nil
        },
    }
    if binary {
        err = embedder.EmbedBin(istrm)
    } else {
        err = embedder.EmbedText(istrm)
    }
    if err == io.EOF {
        err = nil
    }
    return
}

func (eb embeddings) placeholder() (rtn mat.Vector) {
    _, k := eb.Dims()
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

func coocs(documents corpus, eb embeddings, n int, placeholder mat.Vector, progress func(float64)) (J mat.Matrix, F mat.Vector) {
    v, _ := eb.Dims()
    juxt := sparse.NewDOK(v, v)
    freqs := make([]float64, v)

    for i, doc := range documents {
        if progress != nil {
            progress(float64(i) / float64(len(documents)-1))
        }
        doc.nGrams(n, func(ctx document) {
            incr := 1 / float64(n)
            for _, t := range ctx {
                i, ok := eb.Vocab[t]
                if !ok {
                    continue
                }
                freqs[i] += incr
                for _, w := range ctx {
                    if j, ok := eb.Vocab[w]; ok {
                        juxt.Set(i, j, juxt.At(i, j)+incr)
                    }
                }
            }
        })
    }
    J = juxt
    F = mat.NewVecDense(v, freqs)
    return
}

type document []string

type corpus []document

func strip(src string) string {
    strip := func(r rune) bool {
        return unicode.Is(unicode.Mn, r)
    }
    t := transform.Chain(norm.NFD, transform.RemoveFunc(strip), norm.NFC)
    stripped, _, _ := transform.String(t, src)
    return strings.ToLower(stripped)
}

func tokenize(src string) document {
    return document(strings.Fields(strip(src)))
}

func (doc document) nGrams(n int, forEach func(document)) {
    k := len(doc) - n + 1
    for i := 0; i < k; i++ {
        forEach(doc[i : i+n])
    }
}

func loadCorpus(istrm io.ReadSeeker, progress func(float64)) (C corpus, err error) {
    strmLength, err := istrm.Seek(0, io.SeekEnd)
    bytesRead := int64(0)
    if err != nil {
        return
    }
    if _, err = istrm.Seek(0, io.SeekStart); err != nil {
        return
    }
    rd := bufio.NewReader(istrm)
    paragraph := ""
    buf := strings.Builder{}
    srcs := make(chan string)
    sink := make(chan document)
    var wg sync.WaitGroup
    wg.Add(4)
    for i := 1; i <= 4; i++ {
        go func() {
            defer wg.Done()
            for s := range srcs {
                sink <- tokenize(s)
            }
        }()
    }
    go func() {
        for doc := range sink {
            C = append(C, doc)
        }
    }()
    for ; err == nil; paragraph, err = rd.ReadString('\n') {
        paragraph = strings.TrimSpace(paragraph)
        if paragraph == "" {
            srcs <- buf.String()
            if progress != nil {
                if bytesRead, err = istrm.Seek(0, io.SeekCurrent); err != nil {
                    return
                }
                if progress != nil {
                    progress(float64(bytesRead) / float64(strmLength))
                }
            }
            buf.Reset()
        } else {
            buf.WriteString(paragraph)
            buf.WriteByte('\n')
        }
    }
    close(srcs)
    wg.Wait()
    close(sink)
    if err == io.EOF {
        err = nil
    }
    return
}

// http://cowlet.org/2016/08/23/linear-regression-in-rust.html
func linearRegression(X, Y mat.Matrix) *mat.Dense {
    var svd mat.SVD
    svd.Factorize(X, mat.SVDFull)
    var U, V, A mat.Dense
    svd.UTo(&U)
    svd.VTo(&V)
    A.Mul(U.T(), Y)
    _, orderp1 := X.Dims()
    s := svd.Values(nil)[:orderp1]
    raw := make([]float64, orderp1)
    for i := 0; i < orderp1; i++ {
        raw[i] = A.At(i, 0) / s[i]
    }
    sInvA := mat.NewVecDense(orderp1, raw)
    V.Mul(&V, sInvA)
    return &V
}

// https://github.com/NLPrinceton/ALaCarte/blob/master/compute.py
// wordCoocs: V × V
// embeddings: V × d
// wordFreqs: 1 × V
// weights: 1 × V (optional)
func computeTransform(coocs, eb mat.Matrix, wordFreqs, wordWeights mat.Vector) *mat.Dense {
    v, d := eb.Dims()
    var selection mask
    selection.resize(v)
    for t := 0; t < wordFreqs.Len() - 1; t++ {
        x := wordFreqs.AtVec(t)
        if wordWeights != nil {
            x *= wordWeights.AtVec(t)
        }
        selection.set(t, x > 0)
    }
    J := sparse.NewDOK(selection.length, selection.length)
    V := mat.NewDense(selection.length, d,
        make([]float64, selection.length * d))
    for i := 0; i < selection.length-1; i++ {
        if selection.get(i) {
            for j := 0; j < d-1; j++ {
                V.Set(i, j, eb.At(i, j))
                J.Set(i, j, coocs.At(i, j))
            }
        }
    }
    var A mat.Dense
    A.Mul(J, V)
    for k := 0; k < v - 1; k++ {
        row := A.RawRowView(k)
        for i := range row {
            row[i] /= wordFreqs.AtVec(k)
            if wordWeights != nil {
                row[i] *= wordWeights.AtVec(k)
            }
        }
    }
    return linearRegression(&A, eb)
}

func persistTransform(ostrm io.Writer, A mat.Matrix) (err error) {
    m, n := A.Dims()
    if _, err = fmt.Fprintf(ostrm, "%d %d\n", m, n); err != nil {
        return
    }
    for i := 0; i < m-1; i++ {
        for j := 0; j < n-1; j++ {
            if err = binary.Write(ostrm, binary.LittleEndian, A.At(i, j)); err != nil {
                return
            }
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
    istrm, err := os.Open(embeddingPath)
    if err != nil {
        panic(err)
    }
    binary := false
    if len(embeddingPath) >= 4 && embeddingPath[len(embeddingPath)-4:] == ".bin" {
        binary = true
    }
    eb, err := loadEmbeddings(istrm, binary, pbar("Load embeddings"))
    fmt.Println()
    if err != nil {
        panic(err)
    }
    istrm, err = os.Open(corpusPath)
    if err != nil {
        panic(err)
    }
    corpus, err := loadCorpus(istrm, pbar("Load corpus"))
    fmt.Println()
    placeholder := eb.placeholder()
    J, F := coocs(corpus, eb, windowSize, placeholder, pbar("Evaluate cooccurrences"))
    fmt.Println()
    ostrm, err := os.OpenFile(opath, os.O_CREATE|os.O_WRONLY, 0666)
    if err != nil {
        panic(err)
    }
    fmt.Println("Compute induction matrix...")
    A := computeTransform(J, eb, F, nil)
    fmt.Printf("Persist induction matrix (%s)...\n", opath)
    if err = persistTransform(ostrm, A); err != nil {
        panic(err)
    }
    samples := make([]float64, 0, 1024)
    prnProgress := pbar("Evaluate induced embeddings")
    for i, doc := range corpus {
        doc.nGrams(windowSize, func(ctx document) {
            v := eb.embed(ctx[len(ctx)/2])
            if v == nil {
                return
            }
            _, d := eb.Dims()
            vAvg := mat.NewVecDense(d, make([]float64, d))
            s := 1 / float64(len(ctx))
            for _, t := range ctx {
                v := eb.embed(t)
                if v == nil {
                    v = placeholder
                }
                vAvg.AddScaledVec(vAvg, s, v)
            }
            vHat := mat.NewDense(1, d, make([]float64, d))
            vHat.Mul(A, vAvg.T())
            samples = append(samples, cos(vHat.RowView(0), v))
            prnProgress(float64(i) / float64(len(corpus)-1))
        })
    }
    sort.Slice(samples, func(i, j int) bool {
        return samples[i] < samples[j]
    })
    median := samples[len(samples)/2]
    var sigma float64
    for _, x := range samples {
        r := median - x
        sigma += r * r
    }
    sigma = math.Sqrt(sigma)
    fmt.Printf("μ = %f; σ = %f\n", median, sigma)
}
