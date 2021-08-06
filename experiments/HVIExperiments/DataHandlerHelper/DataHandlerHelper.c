/* DataHandlerHelper.c
//  
//
//  Created by Josie Meyer on 9/7/18.
// All values must be declared static so they don't leak into Python!
// Notes: in order to save memory, we are currently limited to 12 digitizer channels and 5 queues per channel
// But feel free to change constants! */

/* Note: if you need to recompile, navigate to the enclosing folder and put the following line on the command line:
gcc DataHandlerHelper.c -o DataHandlerHelper.o -Wall -I C:\Users\slab\anaconda3\pkgs\python-3.6.5-h0c2934d_0\include -I C:\_Lib\SD1\Libraries\include\c -I C:\_Lib\C\pthreads-win32 -I C:\Users\slab\Anaconda3\Lib\site-packages\numpy\core\include -I C:\_Lib\SD1\Libraries\include\common
*/
#include "DataHandlerHelper.h"

//Constants
static const int QUERY_PAUSE = 1;
static const int MAX_QUEUES_PER_CHANNEL = 6;
static const int INITIAL_NUM_CHANNELS = 4;
static const int HASH_BUCKETS = 30;
static const int DEFAULT_BUCKET_SIZE = 2;
static const int START_NUM_BUFFERS = 10; //Number of buffers to load to each channel at start

//Error codes
static const int LOCK_INITIALIZATION_ERROR = -9000;
static const int LOCK_ERROR = -9001;
static const int UNLOCK_ERROR = -9002;
static const int LOCK_DESTRUCTION_ERROR = -9003;
static const int TIMEOUT_ERROR = -9004;
static const int INVALID_READER_NUMBER = -9005;
static const int MEMORY_ERROR = -9006;
static const int QUEUE_NOT_CREATED = -9007;

//Docstrings to export to Python
static const char module_docstring[] =
"This module provides methods for interfacing Python with the Keysight M31xxA series digitizers' native C code";
static const char configureBufferPool_docstring[] = "Configures the buffer pool.";
static const char getData_docstring[] = "Gets data from the internal queue.";
static const char cleanup_docstring[] = "Cleans up the C memory and releases resources from the instrument.";

//Exported methods dictionary; ensures Python can see these methods
static PyMethodDef module_methods[] = { //Communicates with Python re: the structure of the module
    {"configureBufferPool", DataHandlerHelper_configureBufferPool, METH_VARARGS, configureBufferPool_docstring},
    {"getData", DataHandlerHelper_getData, METH_VARARGS, getData_docstring},
    {"cleanup", DataHandlerHelper_cleanup, METH_VARARGS, cleanup_docstring},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef dhhmodule = {
	PyModuleDef_HEAD_INIT,
	"datahandlerhelper",
	module_docstring,
	-1,
	module_methods
};

PyMODINIT_FUNC //Causes Python to recognize as module
PyInit_datahandlerhelper(void)
{
    PyObject *m;
	import_array();
    m = PyModule_Create(&dhhmodule);
    if (m == NULL)
        return NULL;
    return m;
}

/* Global variables accessible within the C code */
static pthread_mutex_t lock; //Lock that will ensure our access to the data is threadsafe
static bool initialized = 0; //We haven't configured the lock yet
static HashBucket* hash_table; //For looking up modules

static PyObject * DataHandlerHelper_configureBufferPool(PyObject * self, PyObject * args)
{
    /* Configures the buffer pool. Intended to be called from Python code.
     Params:
        int module_ID: The module handle that distinguishes the module.
        int channel_number: The number of the channel
        int num_points: The number of data points to collect at one time
        int timeout: Wait time for data, or 0 for infinite.
        int num_trials: The number of trials for which the experiment will run.
     Returns an error code if any, or NULL in case of argument error */
     
    int module_ID, channel_number, num_points, timeout, num_trials, err, i;
	short* init_buffer;
	Queue* queue;
	void* queue_as_void;
    
    //Parse arguments from python function
    if (!PyArg_ParseTuple(args, "iiiii", &module_ID, &channel_number, &num_points, &timeout, &num_trials)) {
        PyErr_SetString(PyExc_TypeError, "Illegal argument, must be 5 ints");
        return (PyObject*) NULL; //Signals erro+r if argument not correct
    }
    
    //Build queue
    err = createQueue(num_trials, num_points, module_ID, channel_number, queue);
	queue_as_void = (void*) queue;
    // Return any error
    if (err < 0) return Py_BuildValue("i", err);
    
    //Allocate memory for first buffer. No need to keep track of pointer as we will get this memory back later.
    init_buffer = (short*) malloc(num_points * sizeof(short));
    if (init_buffer == NULL) return Py_BuildValue("i", MEMORY_ERROR);
    
    //Call the C library function, and pass in callback
    err = SD_AIN_DAQbufferPoolConfig(module_ID, channel_number, init_buffer, num_points, timeout,
                                     callback, queue_as_void); //Hello Ryan! Here's where I'm trying to pass in the callback
    if (err < 0) { //Library function returned an error
        return Py_BuildValue("i", err);
    }
    
    //Add more buffers
    for (i = 0; i < START_NUM_BUFFERS - 1; i++) {
        err = addBuffer(module_ID, channel_number, num_points);
        if (err < 0) return Py_BuildValue("i", err);
    }
    return Py_BuildValue("i", 0); //No error
}

static int put(short* new_data, Queue* queue) {
    /* Puts a data array into the queue, and adds another buffer to buffer pool.
     Params:
     queue_as_void: The queue into which to put the data. Passed in as void* for type correctness with callback
     new_data: The data buffer to add to the queue
     Returns: 0 for successful operation, -9001 for lock error, -9002 for unlock error,
     -9003 for malloc fail, native error code*/
    
    int slot; //Our slot in the queue. We retain it even if interrupted by another call to put
    if (pthread_mutex_lock(&lock)) { //Wait for unlock and check for errors
        PyErr_SetString(PyExc_RuntimeError, "Error locking mutex");
        return LOCK_ERROR; //Signal error if it returns -1 instead of 0
    }
    slot = ++(queue->last_claimed);
    if (pthread_mutex_unlock(&lock)) {
        PyErr_SetString(PyExc_RuntimeError, "Error unlocking mutex");
        return UNLOCK_ERROR; //Signal error on unlock
    }
    queue->data[slot] = new_data;
    queue->ready_to_read[slot] = true;
    return 0; //No error
}

/* Hello Ryan! This is the method that need to be called when the data is ready. As you can see, it currently takes
as parameters the new data and a void pointer that corresponds to the queue that the data is to be put in (passed in
as void* callbackUserObj into the function SD_AIN_DAQbufferPoolConfig, in DataHandlerHelper_configureBufferPool) */

/*void callback(short* new_data, void* queue_as_void) { */
	/* Function to be called when data is ready.
	Params:
	new_data: A buffer containing the new data.
	queue_as_void: A void pointer pointing to the Queue struct where the data is to be put. Queue struct stores
	the length of the buffer among other useful parameters. */ /*
	Queue* queue;
	queue = (Queue*) queue_as_void;
	addBuffer(queue->module_ID, queue->channel_number, queue->points_per_trial);
	// Add a buffer back so pool isn't depleted 
	put(data, queue); // Put the data in the queue
} */

void callback(void* SDobject, int eventNumber, void* buffer, int numData, void* buffer2, int numData2, void *userObject,
int status) {
	short* dataBuffer;
	Queue* queue;
	dataBuffer = (short*) buffer;
	queue = (Queue*) userObject;
	addBuffer(queue->module_ID, queue->channel_number, queue->points_per_trial);
	/*Add a buffer back so pool isn't depleted */
	put(dataBuffer, queue);
}

static int get(Queue* queue, int identifier, int num_tries, volatile short* data) {
    /* Gets a value from the specified queue.
     Params:
     queue: The queue from which to get the data
     identifier: The identifier of the specific reader accessing the queue
     num_tries: The number of tries before timing out.
     data: An outparameter containing a reference to the data
     Returns: 0, or an error code */
    int i;
	int next;
	volatile bool* ready;
	
    if (identifier >= MAX_QUEUES_PER_CHANNEL) return INVALID_READER_NUMBER;
    
    next = ++((queue->queue_positions)[identifier]);
    ready = &(queue->ready_to_read[next]);
    for (i = 0; i < num_tries; ++i) { //Block until timeout or data available
        if (ready) break;
        Sleep(QUERY_PAUSE);
        if (i == num_tries) return TIMEOUT_ERROR; //We timed out
    }
    data = (queue->data)[next];
    return 0;
}

static int addBuffer(int module_ID, int channel_number, int num_points) {
    /* Creates and adds a buffer to the specified buffer pool
     Params:
     module_ID: The module ID of the channel in question
     channel_number: The number of the channel for whom the buffer is being added
     num_points: The length of the buffer to add
     Returns: any error codes */
	short* new_buffer;
    new_buffer = (short*) malloc(num_points * sizeof(short));
    if (new_buffer == NULL) {
        return MEMORY_ERROR; //Signal error on malloc
    }
    return SD_AIN_DAQbufferAdd(module_ID, channel_number, new_buffer, num_points);
}

static unsigned int hashCode(int module_ID, int channel_number) {
    /* Hash code for queue, based on its two unique values -- module_ID and channel_number.
     Taken from internet; yes I know it contains magic numbers! */
	 unsigned int mod, ch;
	 mod = (unsigned int) module_ID;
	 ch = (unsigned int) channel_number;
    return (ch << 19 | mod << 7);
}

static int initialSetup() {
    /* Performs the initial setup of the queues and initialization of variables.
     Returns: 0, or an error code */
	int i, err;
	HashBucket* bucket;
	Queue** entries;
	
    hash_table = malloc(HASH_BUCKETS * sizeof(HashBucket));
    if (hash_table == NULL) {
        return MEMORY_ERROR;
    }
    for (i = 0; i < HASH_BUCKETS; i++) {
        entries = (Queue**) malloc(DEFAULT_BUCKET_SIZE * sizeof(Queue*));
        if (entries == NULL) return MEMORY_ERROR;
        bucket = &hash_table[i];
        bucket->entries = entries;
        bucket->capacity = DEFAULT_BUCKET_SIZE;
        bucket->num_filled = 0;
    }
    err = pthread_mutex_init(&lock, NULL);
    if (err != 0) return LOCK_INITIALIZATION_ERROR;
    initialized = true;
    return 0;
}

static int createQueue(int num_trials, int points_per_trial, int module_ID, int channel_number, Queue* queue) {
    /* Creates a queue and sets it up.
     Params:
     num_trials: The number of trials for which the queue will take data.
     points_per_trial: The number of data points the queue expects for each trial.
     module_ID: The ID number of the module to which the queue is assigned.
     channel_number: The number of the channel to which the queue is assigned.
     queue: An outparameter containing the created queue.
     Returns: 0, or any error codes*/
    
    //Initialize entire module if not already
	int err, i, j;
	volatile short** data;
	bool* ready_to_read;
	
    if (! initialized) {
        err = initialSetup();
        if (err < 0) return err;
    }
    
    //Create and set up queue
    queue = (Queue*) malloc(sizeof(Queue));
    data = (volatile short**) malloc(num_trials * sizeof(short*));
    ready_to_read = (bool*) malloc(num_trials * sizeof(bool));
    if (queue == NULL || data == NULL || ready_to_read == NULL) return MEMORY_ERROR;
    queue->data = data;
    queue->num_trials = num_trials;
    queue->points_per_trial = points_per_trial;
    queue->last_claimed = -1;
	queue->queue_positions = (int*) malloc(MAX_QUEUES_PER_CHANNEL * sizeof(int));
    for (i = 0; i < MAX_QUEUES_PER_CHANNEL; i++) {
        queue->queue_positions[i] = -1;
    }
    for (j = 0; j < num_trials; j++) {
        ready_to_read[j] = false;
    }
    queue->ready_to_read = ready_to_read;
    queue->module_ID = module_ID;
    queue->channel_number = channel_number;
    
    //Add queue to hash table
    return addToHashTable(queue);
}

static int addToHashTable(Queue* queue) {
    /* Adds the current queue to the global hash table to be gotten later
     Params:
     queue: The queue to add to the hash table.
     Returns: 0, or any error codes */
    HashBucket* bucket;
	int num_filled, i, j;
	Queue** new_list;

    bucket = &(hash_table[hashCode(queue->module_ID, queue->channel_number) % HASH_BUCKETS]);
    
    if (bucket->num_filled == bucket->capacity) {//Need to grow bucket
        num_filled = bucket->num_filled;
        new_list = (Queue**) malloc(2 * num_filled * sizeof(Queue*));
        if (new_list == NULL) return MEMORY_ERROR;
        for (i = 0; i < num_filled; i++) {
            new_list[i] = bucket->entries[i];
        }
        for (j = num_filled; j < 2 * num_filled; j++) {
            new_list[j] = 0;
        }
        free(bucket->entries);
        bucket->entries = new_list;
        (bucket->capacity *= 2);
    }
    ++(bucket->num_filled);
    (bucket->entries)[bucket->num_filled] = queue;
    return 0;
}

static int getFromHashTable(int module_ID, int channel_number, Queue* queue) {
    /* Gets a queue from the hash table.
     Params:
     module_ID: The identifier of the module to which the queue is listening for data.
     channel_number: The channel number to which the queue is listening for data.
     queue: An outparameter containing the desired queue if it is found.
     Returns: 0, or any error message */
    HashBucket* bucket;
	int i;
	Queue* possible_queue;
	
	bucket = &(hash_table[hashCode(module_ID, channel_number) % HASH_BUCKETS]);
    for (i = 0; i < (bucket->num_filled); i++) {
        possible_queue = bucket->entries[i];
        if (queue->module_ID == module_ID && queue->channel_number == channel_number) {
            queue = possible_queue;
            return 0;
        }
    }
    return QUEUE_NOT_CREATED;
}

static PyObject* DataHandlerHelper_getData(PyObject* self, PyObject* args) {
    /* Gets the stored data and returns as a numpy array.
     Params:
     module_ID: The ID number of the module.
     channel_number: The number of the channel.
     identifier: The identifier of the data handler that will receive the data. Ensures that multiple data
     handlers are getting the right data.
     timeout: The time, in ms, to delay before returning an error.
     Returns: a tuple with (error message, data or None if an error). */
    int module_ID, channel_number, identifier, err, dimension;
    double timeout;
    short* data;
    Queue* queue;
	PyObject* array;
    
    if (! PyArg_ParseTuple(args, "iiid", &module_ID, &channel_number, &identifier, &timeout)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL; //Signal error
    }
	
    err = getFromHashTable(module_ID, channel_number, queue);
    if (err < 0) {
        Py_INCREF(Py_None);
        return Py_BuildValue("iO", err, Py_None);
    }
	Py_BEGIN_ALLOW_THREADS //Temporarily release GIL so that we are not unnecessarily blocking threads during data retrieval
    err = get(queue, identifier, (int) (timeout / QUERY_PAUSE), data);
	Py_END_ALLOW_THREADS
    if (err < 0) {
        Py_INCREF(Py_None);
        return Py_BuildValue("iO", err, Py_None);
    }
    dimension = 1;
    array = PyArray_SimpleNewFromData(1, (npy_intp*) &dimension, NPY_INT16, data);
    return Py_BuildValue("iO", 0, array);
}

static PyObject* DataHandlerHelper_cleanup(PyObject* self, PyObject* args) {
    int err, i, j;
	
	err = 0;
    if (! PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "Illegal argument, should be empty");
        return NULL;
    }
    //for bucket in hash table
    //for contents if it contains anything:
    //  free buffer pool from modules
    //  free memory for the queue contents
    //  free memory for the queue
    //free memory for the bucket
    //free the hash table
    
    for (i = 0; i < HASH_BUCKETS; i++) { //Iterate over buckets in the hash table
        for (j = 0; j < (hash_table[i].num_filled); j++) {
            err = freeQueue((hash_table[i].entries)[j]);
            if (err < 0) return Py_BuildValue("i", err);
        }
        free(hash_table[i].entries); //Free the list inside the hash bucket
        free(&hash_table[i]); //Free the bucket
    }
    free(hash_table); //Free the hash table itself
    if (pthread_mutex_destroy(&lock)) return Py_BuildValue("i", LOCK_DESTRUCTION_ERROR);
    return Py_BuildValue("i", 0); //Signal no error
}

static int freeQueue(Queue* queue) {
    /* Deletes a queue, given a pointer to the queue. Also deletes all data so make sure it's done being
     processed!
     Returns: Any error code */
    
    //Get any remaining buffers back from the buffer pool and free them
	int err, i;
	short* buffer;
	
    err = SD_AIN_DAQbufferPoolRelease(queue->module_ID, queue->channel_number);
    if (err < 0) return err;
    while (true) {
        buffer = SD_AIN_DAQbufferRemove(queue->module_ID, queue->channel_number);
        if (buffer == NULL) break; //We already got the last buffer
        free(buffer);
    }
    
    for (i = 0; i < queue->num_trials; i++) {
        free((short*) &(queue->data[i]));
    }
    free((short**) (queue->data));
    free((void*) (queue->ready_to_read));
	return 0;
}
