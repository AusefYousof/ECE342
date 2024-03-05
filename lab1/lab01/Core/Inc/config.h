#ifndef __CONFIG_H
#define __CONFIG_H

void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_USART3_UART_Init(void);
void MX_USB_OTG_FS_PCD_Init(void);

void print_msg(char * msg);

#endif