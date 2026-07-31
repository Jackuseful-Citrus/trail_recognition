#ifndef K230_UART_H
#define K230_UART_H

#include "Modules/communication/k230_protocol.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float target_x;
    float target_y;
    float confidence;
    uint32_t sequence;
    uint32_t last_update_ms;
} K230_Target_t;

typedef struct {
    uint32_t rx_events;
    uint32_t rx_bytes;
    uint32_t valid_frames;
    uint32_t unsupported_frames;
    uint32_t crc_errors;
    uint32_t format_errors;
    uint32_t ring_overflows;
    uint32_t uart_errors;
    uint32_t rx_restart_errors;
    uint32_t step_frames;
    uint32_t command_queue_overflows;
    uint32_t status_queue_overflows;
    uint32_t tx_frames;
    uint32_t tx_bytes;
    uint32_t tx_errors;
} K230_UartStats_t;

extern volatile K230_Target_t g_k230_target;
extern volatile K230_UartStats_t g_k230_uart_stats;

bool K230_Uart_Init(void);
bool K230_Uart_GetTarget(K230_Target_t *out);
bool K230_Uart_GetStepCommand(K230_StepCommand_t *out);
bool K230_Uart_SendStepStatus(const K230_StepStatus_t *status);
void K230_Uart_GetStats(K230_UartStats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* K230_UART_H */
