"""Standalone CanMV K230 UART2 continuous-send hardware test.

Upload this file by itself and run it on the K230.  It sends one ASCII test
record every 500 ms; the production vision/planning program is not involved.
"""


UART2_TX_PIN = 5
UART2_RX_PIN = 6
UART2_BAUDRATE = 115200
UART2_SEND_INTERVAL_MS = 500
UART2_TEST_MESSAGE = b"hello\r\n"


def main():
    import time
    from machine import FPIOA, UART

    fpioa = FPIOA()
    fpioa.set_function(UART2_TX_PIN, FPIOA.UART2_TXD)
    fpioa.set_function(UART2_RX_PIN, FPIOA.UART2_RXD)
    uart2 = UART(
        UART.UART2,
        baudrate=UART2_BAUDRATE,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
        timeout=0,
    )
    print(
        "UART2_TEST_READY,tx_pin={},rx_pin={},baudrate={},interval_ms={}".format(
            UART2_TX_PIN,
            UART2_RX_PIN,
            UART2_BAUDRATE,
            UART2_SEND_INTERVAL_MS,
        )
    )

    sequence = 0
    try:
        while True:
            written = uart2.write(UART2_TEST_MESSAGE)
            print(
                "UART2_TEST_SENT,count={},message=hello,bytes={}".format(
                    sequence,
                    (
                        written
                        if written is not None
                        else len(UART2_TEST_MESSAGE)
                    ),
                )
            )
            sequence += 1
            time.sleep_ms(UART2_SEND_INTERVAL_MS)
    finally:
        try:
            uart2.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
