package main

import "github.com/teivah/bitvector"

type mask struct {
    vectors []bitvector.Len64
    length int
}

func (m *mask) vecAt(i int) *bitvector.Len64 {
    vi := i >> 6
    if vi > len(m.vectors)-1 {
        return nil
    }
    return &m.vectors[vi]
}

func (m *mask) capacity() int {
    return len(m.vectors) << 6
}

// resize to fit at least n elements
func (m *mask) resize(capacity int) {
    canFit := m.capacity()
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

func (m *mask) get(i int) bool {
    v := m.vecAt(i)
    if v == nil {
        return false
    }
    return v.Get(uint8(i % 64))
}

func (m *mask) set(i int, p bool) {
    v := m.vecAt(i)
    if v == nil {
        m.resize(i+1)
    }
    if p {
        *m.vecAt(i) = m.vecAt(i).Set(uint8(i%64), p)
    }
    if i+1 > m.length {
        m.length = i+1
    }
}

