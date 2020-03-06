package rw_coordinator

import (
	"net/url"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"gotest.tools/assert"
)

func readValue(t *testing.T, addr string, sleepTime time.Duration, wg *sync.WaitGroup) {
	defer wg.Done()
	u := url.URL{Scheme: "ws", Host: addr, Path: "/resource1", RawQuery: "read_lock=True"}
	c, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	assert.NilError(t, err)
	defer c.Close()

	_, message, err := c.ReadMessage()
	assert.NilError(t, err)
	assert.Equal(t, string(message), "read_lock_granted", "Did not receive `read_lock_granted` response from server: %s", string(message))

	time.Sleep(sleepTime)
}

func writeValue(t *testing.T, addr string, sleepTime time.Duration, wg *sync.WaitGroup, sharedValue *int) {
	defer wg.Done()
	u := url.URL{Scheme: "ws", Host: addr, Path: "/resource1", RawQuery: "read_lock=False"}
	c, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	assert.NilError(t, err)
	defer c.Close()

	_, message, err := c.ReadMessage()
	assert.NilError(t, err)
	assert.Equal(t, string(message), "write_lock_granted", "Did not receive `write_lock_granted` response from server: %s", string(message))

	time.Sleep(sleepTime)
	*sharedValue += 1
}

func stopServer(t *testing.T, addr string) {
	u := url.URL{Scheme: "ws", Host: addr, Path: "/shutdown"}
	_, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	assert.Error(t, err, "websocket: bad handshake")
}

func TestServer(t *testing.T) {
	addr := "localhost:8081"
	numThreads := 4
	sharedValue := 0
	var wg sync.WaitGroup

	go RunServer(addr)

	wg.Add(numThreads * 2)
	for i := 0; i < numThreads; i++ {
		go readValue(t, addr, time.Duration(i)*time.Second, &wg)
		go writeValue(t, addr, time.Duration(i)*time.Second, &wg, &sharedValue)
	}
	wg.Wait()
	assert.Equal(t, sharedValue, numThreads)

	stopServer(t, addr)
}
