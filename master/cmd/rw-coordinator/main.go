package main

import (
	"flag"

	server "github.com/determined-ai/yogadl/master/pkg/rw-coordinator"
)

var addr = flag.String("addr", "localhost:8080", "rw-coordinator server address")

func main() {
	flag.Parse()
	server.RunServer(*addr)
}
