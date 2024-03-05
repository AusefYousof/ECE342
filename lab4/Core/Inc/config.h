#ifndef __CONFIG_H
#define __CONFIG_H

#include "main.h"

extern I2C_HandleTypeDef hi2c1;

extern UART_HandleTypeDef huart3;
extern PCD_HandleTypeDef hpcd_USB_OTG_FS;

void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_I2C1_Init(void);
void MX_USART3_UART_Init(void);
void MX_USB_OTG_FS_PCD_Init(void);

void print_msg(char * msg);

#endif