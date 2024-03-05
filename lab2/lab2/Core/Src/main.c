/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
#include "main.h"
#include "config.h"

#include <stdio.h>
#include <math.h>


int main(void)
{
  /* Reset of all peripherals. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC3_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  MX_DAC_Init();

  uint16_t adc_res, mask = 0xff00;
  char message[100];
  
  //  ADC example
  HAL_ADC_Start(&hadc3);
  HAL_ADC_PollForConversion(&hadc3, 100);
  adc_res = HAL_ADC_GetValue(&hadc3);
  sprintf(message, "adc_res=%d\r\n", adc_res);
  print_msg(message);
  
  // DAC example
  HAL_DAC_Init(&hdac);
  HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
  HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, 255);

  while (1)
  {
		
		//part 2 - sampling with ADC
		//begin ADC
		HAL_ADC_Start(&hadc3);
		
		//block excecution, wait for adc conversion
		HAL_ADC_PollForConversion(&hadc3, 100);
		
		//print to UART
		adc_res = HAL_ADC_GetValue(&hadc3);
		sprintf(message, "adc_res=%d\r\n", adc_res);
		print_msg(message);
		
		
		//part 3, convert to analog
		//scale adc value received to 12 bit for stm32
    uint16_t dacValue = adc_res * 4095 / 4095; 

    // Set the DAC value
    HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacValue);
  }
 }