package rw_coordinator

import (
	"context"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
	log "github.com/sirupsen/logrus"
)

var upgrader = websocket.Upgrader{}

type resourceObject struct {
	mux sync.RWMutex
}

type resourceMap struct {
	resources map[string]*resourceObject
	mux       sync.Mutex
}

func (rMap *resourceMap) getResource(resourceName string) *resourceObject {
	rMap.mux.Lock()
	defer rMap.mux.Unlock()

	if resource, ok := rMap.resources[resourceName]; ok {
		return resource
	}

	rMap.resources[resourceName] = &(resourceObject{})
	return rMap.resources[resourceName]
}

func (rMap *resourceMap) requestHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Info(err)
		return
	}

	resourceName := r.URL.Path
	query := r.URL.Query()

	readLockString, ok := query["read_lock"]
	if !ok {
		log.Error("Received request without specifying read_lock ", r.URL)
	}

	var readLock bool
	if readLockString[0] == "True" {
		readLock = true
	} else {
		readLock = false
	}

	resource := rMap.getResource(resourceName)

	var response string
	if readLock {
		resource.mux.RLock()
		defer resource.mux.RUnlock()
		response = "read_lock_granted"
	} else {
		resource.mux.Lock()
		defer resource.mux.Unlock()
		response = "write_lock_granted"
	}

	if err := conn.WriteMessage(websocket.TextMessage, []byte(response)); err != nil {
		log.Info("rw-coordinator server failed to respond: ", err)
	}

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			log.Info("Connection closed: ", err)
			return
		}
	}
}

// RunServer launches the rw-coordinator server.
func RunServer(address string) {
	rMap := resourceMap{resources: make(map[string]*resourceObject)}
	serverMutex := http.NewServeMux()
	server := http.Server{Addr: address, Handler: serverMutex}
	serverMutex.HandleFunc("/", rMap.requestHandler)
	serverMutex.HandleFunc("/shutdown", func(w http.ResponseWriter, r *http.Request) {
		go func() {
			if err := server.Shutdown(context.Background()); err != nil {
				log.Info("rw-coordinator server shutdown failed: ", err)
			}
		}()
	})
	if err := server.ListenAndServe(); err != nil {
		log.Info("rw-coordinator server stopped: ", err)
	}
}
