package mask

import "github.com/teivah/bitvector"

type Mask struct {
    vectors []bitvector.Len64
    length int
}

func (m *Mask) VecAt(i int) *bitvector.Len64 {
    vi := i >> 6
    if vi > len(m.vectors)-1 {
        return nil
    }
    return &m.vectors[vi]
}

func (m *Mask) Cap() int {
    return len(m.vectors) << 6
}

func (m *Mask) Len() int {
    return m.length
}

// resize to fit at least n elements
func (m *Mask) Resize(capacity int) {
    canFit := m.Cap()
    shouldFit := canFit + capacity
    growElem := shouldFit - canFit
    growVecs := growElem >> 6
    if growElem % 64 != 0 {
        growVecs++
    }
    growVecs -= len(m.vectors)
    if growVecs <= 0 {
        return
    }
    m.vectors = append(m.vectors, make([]bitvector.Len64, growVecs)...)
}

func (m *Mask) Get(i int) bool {
    v := m.VecAt(i)
    if v == nil {
        return false
    }
    return v.Get(uint8(i % 64))
}

func (m *Mask) Set(i int, p bool) {
    v := m.VecAt(i)
    if v == nil {
        m.Resize(i+1)
    }
    if p {
        *m.VecAt(i) = m.VecAt(i).Set(uint8(i%64), p)
    }
    if i+1 > m.length {
        m.length = i+1
    }
}

