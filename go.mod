module github.com/kavorite/induction

replace github.com/kavorite/induction/word2vec => ./word2vec

require (
	github.com/deckarep/golang-set v1.7.1 // indirect
	github.com/gonum/floats v0.0.0-20181209220543-c233463c7e82 // indirect
	github.com/gonum/internal v0.0.0-20181124074243-f884aa714029 // indirect
	github.com/james-bowman/sparse v0.0.0-20191203082507-0ce573ca6b71
	github.com/kavorite/induction/word2vec v0.0.0-00010101000000-000000000000
	github.com/mingrammer/commonregex v1.0.1 // indirect
	gonum.org/v1/gonum v0.6.1
	gopkg.in/jdkato/prose.v2 v2.0.0-20190814032740-822d591a158c
	gopkg.in/neurosnap/sentences.v1 v1.0.6 // indirect
)

go 1.13
