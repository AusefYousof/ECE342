#ifndef __CONFIG_H
#define __CONFIG_H

#include "main.h"

extern ADC_HandleTypeDef hadc3;
extern DAC_HandleTypeDef hdac;


void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_ADC3_Init(void);
void MX_USART3_UART_Init(void);
void MX_USB_OTG_FS_PCD_Init(void);
void MX_DAC_Init(void);

void print_msg(char * msg);

#endif