package word2vec

import (
	"encoding/binary"
	"fmt"
	"io"
)

type Embedder struct {
	Head  func(wordc, dimen int) error
	Embed func(token string, embeddings []float32) error
}

func (cfg Embedder) EmbedBin(istrm io.Reader) (err error) {
	var (
		wordc, dimen int
		t            string
	)
	_, err = fmt.Fscanf(istrm, "%d %d", &wordc, &dimen)
	if err != nil {
		return
	}
	if cfg.Head != nil {
		if err = cfg.Head(wordc, dimen); err != nil {
			return
		}
	}
	for i := 0; i < wordc; i++ {
		_, err = fmt.Fscanf(istrm, "%s", &t)
		if err != nil {
            if err == io.EOF {
                err = nil
            }
			return
		}
		embedding := make([]float32, dimen)
		if err = binary.Read(istrm, binary.LittleEndian, embedding); err != nil {
			return
		}
		if cfg.Embed == nil {
			continue
		}
		if err = cfg.Embed(t, embedding); err != nil {
			return
		}
	}
	return
}

func (cfg Embedder) EmbedText(istrm io.Reader) (err error) {
	var (
		wordc, dimen int
		t            string
	)
	_, err = fmt.Fscanf(istrm, "%d %d", &wordc, &dimen)
	if err != nil {
		return
	}
	if cfg.Head != nil {
		if err = cfg.Head(wordc, dimen); err != nil {
			return
		}
	}
	for i := 0; i < wordc; i++ {
		_, err = fmt.Fscanf(istrm, "%s", &t)
        if err != nil {
            if err == io.EOF {
                err = nil
            }
            return
        }
		embedding := make([]float32, dimen)
		for b := 0; b < dimen; b++ {
			_, err = fmt.Fscanf(istrm, "%f", &embedding[b])
			if err != nil {
				return
			}
		}
		if cfg.Embed == nil {
			continue
		}
	}
	return
}
