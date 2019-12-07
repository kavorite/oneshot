module github.com/kavorite/oneshot

go 1.13

replace (
	github.com/kavorite/oneshot/mask => ./mask
	github.com/kavorite/oneshot/word2vec => ./word2vec
)

require (
	github.com/gonum/floats v0.0.0-20181209220543-c233463c7e82 // indirect
	github.com/gonum/internal v0.0.0-20181124074243-f884aa714029 // indirect
	github.com/james-bowman/sparse v0.0.0-20191203082507-0ce573ca6b71
	github.com/kavorite/oneshot/word2vec v0.0.0-00010101000000-000000000000
	github.com/teivah/bitvector v0.0.0-20190716215529-286a776f46ca
	golang.org/x/text v0.3.3-0.20191122225017-cbf43d21aaeb
	gonum.org/v1/gonum v0.6.1
)
