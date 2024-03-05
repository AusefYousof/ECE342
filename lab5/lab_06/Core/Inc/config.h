#ifndef __CONFIG_H
#define __CONFIG_H

#include "main.h"

extern DCMI_HandleTypeDef hdcmi;
extern DMA_HandleTypeDef hdma_dcmi;

extern I2C_HandleTypeDef hi2c2;

extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim6;

extern UART_HandleTypeDef huart3;

extern PCD_HandleTypeDef hpcd_USB_OTG_FS;


HAL_StatusTypeDef print_msg(char * msg);

HAL_StatusTypeDef uart_send_bin(uint8_t * buff, unsigned int len);

void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_DMA_Init(void);
void MX_DCMI_Init(void);
void MX_USART3_UART_Init(void);
void MX_USB_OTG_FS_PCD_Init(void);
void MX_I2C2_Init(void);
void MX_TIM1_Init(void);
void MX_TIM6_Init(void);

#endif
