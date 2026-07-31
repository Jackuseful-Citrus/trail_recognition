#include "Modules/communication/k230_uart.h"

#include "cmsis_os2.h"
#include "usart.h"

#include <string.h>

#define K230_DMA_RX_SIZE 256U
#define K230_RING_SIZE 512U
#define K230_PAYLOAD_MAX_SIZE 32U
#define K230_FRAME_MAX_SIZE (2U + 1U + 1U + K230_PAYLOAD_MAX_SIZE + 2U)
#define K230_STEP_QUEUE_DEPTH 4U
#define K230_STATUS_QUEUE_DEPTH 8U
#define K230_STATUS_FRAME_SIZE (2U + 1U + 1U + K230_STEP_STATUS_PAYLOAD_SIZE + 2U)

#define K230_HEADER_LSB 0x55U
#define K230_HEADER_MSB 0xAAU

volatile K230_Target_t g_k230_target;
volatile K230_UartStats_t g_k230_uart_stats;

static uint8_t s_dma_rx_buffer[K230_DMA_RX_SIZE]
    __attribute__((section(".dma_buffer"), aligned(32)));
static uint8_t s_ring_buffer[K230_RING_SIZE];
static volatile uint16_t s_ring_head;
static volatile uint16_t s_ring_tail;
static volatile uint16_t s_dma_last_pos;
static volatile bool s_initialized;
static osMessageQueueId_t s_step_queue;
static osMessageQueueId_t s_status_queue;

static uint8_t s_frame[K230_FRAME_MAX_SIZE];
static uint8_t s_frame_len;
static uint8_t s_expected_frame_len;

static uint16_t k230_next_index(uint16_t value, uint16_t size) {
    ++value;
    return (value >= size) ? 0U : value;
}

static bool k230_ring_write(uint8_t byte) {
    const uint16_t head = s_ring_head;
    const uint16_t next = k230_next_index(head, K230_RING_SIZE);
    if (next == s_ring_tail) {
        ++g_k230_uart_stats.ring_overflows;
        return false;
    }

    s_ring_buffer[head] = byte;
    s_ring_head = next;
    return true;
}

static bool k230_ring_read(uint8_t *byte) {
    const uint16_t tail = s_ring_tail;
    if (tail == s_ring_head) {
        return false;
    }

    *byte = s_ring_buffer[tail];
    s_ring_tail = k230_next_index(tail, K230_RING_SIZE);
    return true;
}

static uint16_t k230_crc16_ccitt_false(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFFU;

    for (uint16_t i = 0U; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0U; bit < 8U; ++bit) {
            if ((crc & 0x8000U) != 0U) {
                crc = (uint16_t)((crc << 1) ^ 0x1021U);
            } else {
                crc <<= 1;
            }
        }
    }

    return crc;
}

static void k230_reset_parser(void) {
    s_frame_len = 0U;
    s_expected_frame_len = 0U;
}

static void k230_update_target(const uint8_t *payload) {
    float x;
    float y;
    float confidence;
    memcpy(&x, &payload[0], sizeof(x));
    memcpy(&y, &payload[4], sizeof(y));
    memcpy(&confidence, &payload[8], sizeof(confidence));

    uint32_t next_sequence = g_k230_target.sequence + 2U;
    if ((next_sequence & 1U) != 0U || next_sequence == 0U) {
        next_sequence = 2U;
    }

    g_k230_target.sequence = next_sequence | 1U;
    __DMB();
    g_k230_target.target_x = x;
    g_k230_target.target_y = y;
    g_k230_target.confidence = confidence;
    g_k230_target.last_update_ms = HAL_GetTick();
    __DMB();
    g_k230_target.sequence = next_sequence;
}

static void k230_queue_step(const uint8_t *payload) {
    K230_StepCommand_t command = {
        .version = payload[0],
        .sequence = payload[1],
        .action = payload[2],
        .flags = payload[3],
        .value = 0.0f,
    };
    memcpy(&command.value, &payload[4], sizeof(command.value));

    if (s_step_queue == NULL ||
        osMessageQueuePut(s_step_queue, &command, 0U, 0U) != osOK) {
        ++g_k230_uart_stats.command_queue_overflows;
        return;
    }
    ++g_k230_uart_stats.step_frames;
}

static void k230_handle_frame(void) {
    const uint8_t command = s_frame[2];
    const uint8_t payload_len = s_frame[3];
    const uint16_t crc_offset = (uint16_t)(s_expected_frame_len - 2U);
    const uint16_t received_crc = (uint16_t)s_frame[crc_offset] |
                                  ((uint16_t)s_frame[crc_offset + 1U] << 8);
    const uint16_t calculated_crc =
        k230_crc16_ccitt_false(&s_frame[2], (uint16_t)(2U + payload_len));

    if (received_crc != calculated_crc) {
        ++g_k230_uart_stats.crc_errors;
        return;
    }

    ++g_k230_uart_stats.valid_frames;
    if (command == K230_CMD_TARGET_LEGACY && payload_len == K230_TARGET_PAYLOAD_SIZE) {
        k230_update_target(&s_frame[4]);
    } else if (command == K230_CMD_STEP && payload_len == K230_STEP_PAYLOAD_SIZE) {
        k230_queue_step(&s_frame[4]);
    } else {
        ++g_k230_uart_stats.unsupported_frames;
    }
}

static void k230_parse_byte(uint8_t byte) {
    if (s_frame_len == 0U) {
        if (byte == K230_HEADER_LSB) {
            s_frame[s_frame_len++] = byte;
        }
        return;
    }

    if (s_frame_len == 1U) {
        if (byte == K230_HEADER_MSB) {
            s_frame[s_frame_len++] = byte;
        } else if (byte != K230_HEADER_LSB) {
            k230_reset_parser();
        }
        return;
    }

    if (s_frame_len >= K230_FRAME_MAX_SIZE) {
        ++g_k230_uart_stats.format_errors;
        k230_reset_parser();
        return;
    }

    s_frame[s_frame_len++] = byte;

    if (s_frame_len == 4U) {
        const uint8_t payload_len = s_frame[3];
        if (payload_len > K230_PAYLOAD_MAX_SIZE) {
            ++g_k230_uart_stats.format_errors;
            k230_reset_parser();
            return;
        }
        s_expected_frame_len = (uint8_t)(2U + 1U + 1U + payload_len + 2U);
    }

    if (s_expected_frame_len != 0U && s_frame_len >= s_expected_frame_len) {
        k230_handle_frame();
        k230_reset_parser();
    }
}

static void k230_capture_dma_range(uint16_t begin, uint16_t end) {
    for (uint16_t pos = begin; pos < end; ++pos) {
        ++g_k230_uart_stats.rx_bytes;
        (void)k230_ring_write(s_dma_rx_buffer[pos]);
    }
}

static void k230_capture_dma(uint16_t dma_pos) {
    if (dma_pos > K230_DMA_RX_SIZE) {
        ++g_k230_uart_stats.format_errors;
        return;
    }

    uint16_t last_pos = s_dma_last_pos;
    if (dma_pos == last_pos) {
        return;
    }

    if (dma_pos > last_pos) {
        k230_capture_dma_range(last_pos, dma_pos);
    } else {
        k230_capture_dma_range(last_pos, K230_DMA_RX_SIZE);
        if (dma_pos > 0U) {
            k230_capture_dma_range(0U, dma_pos);
        }
    }

    s_dma_last_pos = (dma_pos == K230_DMA_RX_SIZE) ? 0U : dma_pos;
}

static bool k230_start_rx(void) {
    s_dma_last_pos = 0U;
    if (HAL_UARTEx_ReceiveToIdle_DMA(&huart7, s_dma_rx_buffer, K230_DMA_RX_SIZE) != HAL_OK) {
        ++g_k230_uart_stats.rx_restart_errors;
        return false;
    }
    if (huart7.hdmarx != NULL) {
        __HAL_DMA_DISABLE_IT(huart7.hdmarx, DMA_IT_HT);
    }
    return true;
}

static bool k230_transmit_status(const K230_StepStatus_t *status) {
    uint8_t frame[K230_STATUS_FRAME_SIZE] = {
        K230_HEADER_LSB,
        K230_HEADER_MSB,
        K230_CMD_STEP_STATUS,
        K230_STEP_STATUS_PAYLOAD_SIZE,
        status->version,
        status->sequence,
        status->action,
        status->status,
        status->error_code,
        0U,
        0U,
        0U,
        0U,
        0U,
    };
    const uint16_t crc =
        k230_crc16_ccitt_false(&frame[2], 2U + K230_STEP_STATUS_PAYLOAD_SIZE);
    frame[K230_STATUS_FRAME_SIZE - 2U] = (uint8_t)crc;
    frame[K230_STATUS_FRAME_SIZE - 1U] = (uint8_t)(crc >> 8U);

    if (HAL_UART_Transmit(&huart7, frame, sizeof(frame), 20U) != HAL_OK) {
        ++g_k230_uart_stats.tx_errors;
        return false;
    }

    ++g_k230_uart_stats.tx_frames;
    g_k230_uart_stats.tx_bytes += sizeof(frame);
    return true;
}

static void k230_process_status_tx(void) {
    K230_StepStatus_t status;
    while (s_status_queue != NULL &&
           osMessageQueueGet(s_status_queue, &status, NULL, 0U) == osOK) {
        (void)k230_transmit_status(&status);
    }
}

static void k230_uart_task(void *argument) {
    (void)argument;

    for (;;) {
        uint8_t byte;
        while (k230_ring_read(&byte)) {
            k230_parse_byte(byte);
        }
        k230_process_status_tx();
        (void)osDelay(1U);
    }
}

bool K230_Uart_Init(void) {
    if (s_initialized) {
        return true;
    }

    memset((void *)s_dma_rx_buffer, 0, sizeof(s_dma_rx_buffer));
    memset(s_ring_buffer, 0, sizeof(s_ring_buffer));
    k230_reset_parser();
    s_ring_head = 0U;
    s_ring_tail = 0U;
    s_dma_last_pos = 0U;

    g_k230_target.target_x = 0.0f;
    g_k230_target.target_y = 0.0f;
    g_k230_target.confidence = 0.0f;
    g_k230_target.sequence = 0U;
    g_k230_target.last_update_ms = 0U;
    memset((void *)&g_k230_uart_stats, 0, sizeof(g_k230_uart_stats));

    s_step_queue = osMessageQueueNew(K230_STEP_QUEUE_DEPTH,
                                     sizeof(K230_StepCommand_t), NULL);
    s_status_queue = osMessageQueueNew(K230_STATUS_QUEUE_DEPTH,
                                       sizeof(K230_StepStatus_t), NULL);
    if (s_step_queue == NULL || s_status_queue == NULL) {
        return false;
    }

    if (!k230_start_rx()) {
        return false;
    }

    static const osThreadAttr_t task_attributes = {
        .name = "k230_uart",
        .stack_size = 768U,
        .priority = osPriorityNormal,
    };
    if (osThreadNew(k230_uart_task, NULL, &task_attributes) == NULL) {
        return false;
    }

    s_initialized = true;
    return true;
}

bool K230_Uart_GetTarget(K230_Target_t *out) {
    if (out == NULL) {
        return false;
    }

    for (uint8_t attempt = 0U; attempt < 3U; ++attempt) {
        const uint32_t before = g_k230_target.sequence;
        if ((before & 1U) != 0U || before == 0U) {
            continue;
        }

        __DMB();
        out->target_x = g_k230_target.target_x;
        out->target_y = g_k230_target.target_y;
        out->confidence = g_k230_target.confidence;
        out->last_update_ms = g_k230_target.last_update_ms;
        __DMB();
        out->sequence = g_k230_target.sequence;
        if (before == out->sequence) {
            return true;
        }
    }

    return false;
}

bool K230_Uart_GetStepCommand(K230_StepCommand_t *out) {
    if (out == NULL || s_step_queue == NULL) {
        return false;
    }
    return osMessageQueueGet(s_step_queue, out, NULL, 0U) == osOK;
}

bool K230_Uart_SendStepStatus(const K230_StepStatus_t *status) {
    if (status == NULL || s_status_queue == NULL) {
        return false;
    }

    K230_StepStatus_t queued = *status;
    queued.reserved[0] = 0U;
    queued.reserved[1] = 0U;
    queued.reserved[2] = 0U;
    if (osMessageQueuePut(s_status_queue, &queued, 0U, 0U) != osOK) {
        ++g_k230_uart_stats.status_queue_overflows;
        return false;
    }
    return true;
}

void K230_Uart_GetStats(K230_UartStats_t *out) {
    if (out == NULL) {
        return;
    }

    out->rx_events = g_k230_uart_stats.rx_events;
    out->rx_bytes = g_k230_uart_stats.rx_bytes;
    out->valid_frames = g_k230_uart_stats.valid_frames;
    out->unsupported_frames = g_k230_uart_stats.unsupported_frames;
    out->crc_errors = g_k230_uart_stats.crc_errors;
    out->format_errors = g_k230_uart_stats.format_errors;
    out->ring_overflows = g_k230_uart_stats.ring_overflows;
    out->uart_errors = g_k230_uart_stats.uart_errors;
    out->rx_restart_errors = g_k230_uart_stats.rx_restart_errors;
    out->step_frames = g_k230_uart_stats.step_frames;
    out->command_queue_overflows = g_k230_uart_stats.command_queue_overflows;
    out->status_queue_overflows = g_k230_uart_stats.status_queue_overflows;
    out->tx_frames = g_k230_uart_stats.tx_frames;
    out->tx_bytes = g_k230_uart_stats.tx_bytes;
    out->tx_errors = g_k230_uart_stats.tx_errors;
}

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t size) {
    if (huart == &huart7) {
        ++g_k230_uart_stats.rx_events;
        k230_capture_dma(size);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart7) {
        ++g_k230_uart_stats.uart_errors;
        (void)k230_start_rx();
    }
}
