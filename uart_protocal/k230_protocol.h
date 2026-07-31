#ifndef K230_PROTOCOL_H
#define K230_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define K230_PROTOCOL_VERSION          1U
#define K230_PROTOCOL_FLAG_VALID       0x01U

#define K230_CMD_TARGET_LEGACY         0x01U
#define K230_CMD_STEP                  0x10U
#define K230_CMD_STEP_STATUS           0x90U

#define K230_TARGET_PAYLOAD_SIZE       12U
#define K230_STEP_PAYLOAD_SIZE         8U
#define K230_STEP_STATUS_PAYLOAD_SIZE  8U

typedef enum {
    K230_STEP_MOVE_X_ABS = 1,
    K230_STEP_MOVE_Y_ABS = 2,
    K230_STEP_ROTATE_REL = 3,
    K230_STEP_GRIP = 4,
    K230_STEP_Z = 5,
} K230_StepAction_t;

typedef enum {
    K230_STEP_STATUS_BUSY = 0,
    K230_STEP_STATUS_DONE = 1,
    K230_STEP_STATUS_ERROR = 2,
} K230_StepStatusCode_t;

typedef enum {
    K230_STEP_ERROR_NONE = 0,
    K230_STEP_ERROR_BAD_VERSION = 1,
    K230_STEP_ERROR_INVALID_FLAGS = 2,
    K230_STEP_ERROR_UNSUPPORTED_ACTION = 3,
    K230_STEP_ERROR_INVALID_VALUE = 4,
    K230_STEP_ERROR_OUT_OF_RANGE = 5,
    K230_STEP_ERROR_EXECUTOR_BUSY = 6,
    K230_STEP_ERROR_HARDWARE_NOT_READY = 7,
    K230_STEP_ERROR_TIMEOUT = 8,
    K230_STEP_ERROR_SEQUENCE_CONFLICT = 9,
} K230_StepError_t;

typedef struct {
    uint8_t version;
    uint8_t sequence;
    uint8_t action;
    uint8_t flags;
    float value;
} K230_StepCommand_t;

typedef struct {
    uint8_t version;
    uint8_t sequence;
    uint8_t action;
    uint8_t status;
    uint8_t error_code;
    uint8_t reserved[3];
} K230_StepStatus_t;

#ifdef __cplusplus
}
#endif

#endif /* K230_PROTOCOL_H */
