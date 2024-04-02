#ifndef __CONFIG_H
#define __CONFIG_H

#include "main.h"

extern DAC_HandleTypeDef hdac;
extern TIM_HandleTypeDef htim6;
extern USART_HandleTypeDef husart3;

void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_DAC_Init(void);
void MX_TIM6_Init(void);
void MX_USART3_Init(void);
void print_msg(char* msg);

#endif