/* DataHandlerHelper.h.
Created by Josie Meyer, 9/2018 */

#ifndef DataHandlerHelper_h
#define DataHandlerHelper_h

#include "Python.h"

#include <stdlib.h>
#include "SD_AIN.h"
#include <windows.h>
#include <stdbool.h>
#include <pthread.h>
#include <arrayobject.h>


/* Used to build a fast, thread-safe and reentrant-safe queue */
typedef struct queue_struct {
    volatile short** data; /*Holds the actual data */
    int num_trials; /* Number of trials */
    int points_per_trial; /* Number of data points per trial */
    volatile int last_claimed; /* The last slot claimed. Should be initialized to -1. */
    volatile bool* ready_to_read; /* Indicates which data can be read */
    
    volatile int* queue_positions;
    /* Index is the number of the queue in question. Value is the position of reader. */
    
    int module_ID; /* The module_ID of the module associated with the queue */
    int channel_number; /* The channel number */
} Queue;

typedef struct HashBucket_struct { /* Used to make hash table */
    Queue** entries;
    int num_filled;
    int capacity;
} HashBucket;

//Function prototypes
static PyObject* DataHandlerHelper_configureBufferPool(PyObject * self, PyObject * args);
static int put(short* new_data, Queue* queue);
static int get(Queue* queue, int identifier, int num_tries, volatile short* data);
static int addBuffer(int module_ID, int channel_number, int num_points);
static unsigned int hashCode(int module_ID, int channel_number);
static int initialSetup();
static int createQueue(int num_trials, int points_per_trial, int module_ID, int channel_number, Queue* queue);
static int addToHashTable(Queue* queue);
static int getFromHashTable(int module_ID, int channel_number, Queue* queue);
static PyObject* DataHandlerHelper_getData(PyObject* self, PyObject* args);
static PyObject* DataHandlerHelper_cleanup(PyObject* self, PyObject* args);
static int freeQueue(Queue* queue);
void callback(void* SDobject, int eventNumber, void* buffer, int numData, void* buffer2, int numData2, void *userObject,
int status);

#endif /* DataHandlerHelper_h */

