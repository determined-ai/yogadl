GOBIN := $(shell go env GOPATH)/bin

check:
	$(GOBIN)/golangci-lint run

fmt:
	$(GOBIN)/goimports -l -local github.com/determined-ai/yogadl -w .

test:
	go test ./...