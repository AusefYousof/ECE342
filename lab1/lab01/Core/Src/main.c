/* USER CODE BEGIN Header */
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
  *******************************************************************************
  */
#include "main.h"
#include "config.h"
#include <stdio.h>


int8_t current_row = -1, current_col = -1;


int main(void)
{
  /* Reset of all peripherals. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();

  char message[100];
  sprintf(message, "Start..\n");
  print_msg(message);
	
	char pressed[5];

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  
  /* Initialize ROW outputs */
  HAL_GPIO_WritePin(ROW0_GPIO_Port, ROW0_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(ROW1_GPIO_Port, ROW1_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(ROW2_GPIO_Port, ROW2_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(ROW3_GPIO_Port, ROW3_Pin, GPIO_PIN_RESET);

  /* Infinite loop */
  while (1)
  {
		HAL_Delay(80);
		
		//first row 
		HAL_GPIO_WritePin(ROW0_GPIO_Port, ROW0_Pin, GPIO_PIN_SET);
		HAL_GPIO_WritePin(ROW1_GPIO_Port, ROW1_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW2_GPIO_Port, ROW2_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW3_GPIO_Port, ROW3_Pin, GPIO_PIN_RESET);
		
		HAL_Delay(20);
		
		current_row = 1;
		
		if (current_col != -1){
			if (HAL_GPIO_ReadPin(COL0_GPIO_Port, COL0_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 0 is 3 
				sprintf(pressed, "3");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL0_GPIO_Port, COL0_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL1_GPIO_Port, COL1_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 1 is 2 
				sprintf(pressed, "2");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL1_GPIO_Port, COL1_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL2_GPIO_Port, COL2_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "1");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL2_GPIO_Port, COL2_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL3_GPIO_Port, COL3_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "0");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL3_GPIO_Port, COL3_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
		}
		
		//second row
		
		HAL_GPIO_WritePin(ROW0_GPIO_Port, ROW0_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW1_GPIO_Port, ROW1_Pin, GPIO_PIN_SET);
		HAL_GPIO_WritePin(ROW2_GPIO_Port, ROW2_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW3_GPIO_Port, ROW3_Pin, GPIO_PIN_RESET);
				
		current_row = 1;
		
		if (current_col != -1){
			if (HAL_GPIO_ReadPin(COL0_GPIO_Port, COL0_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 1 is 7 
				sprintf(pressed, "7");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL0_GPIO_Port, COL0_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL1_GPIO_Port, COL1_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "6");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL1_GPIO_Port, COL1_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL2_GPIO_Port, COL2_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "5");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL2_GPIO_Port, COL2_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL3_GPIO_Port, COL3_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "4");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL3_GPIO_Port, COL3_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
		}
		
		//third row

		HAL_GPIO_WritePin(ROW0_GPIO_Port, ROW0_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW1_GPIO_Port, ROW1_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW2_GPIO_Port, ROW2_Pin, GPIO_PIN_SET);
		HAL_GPIO_WritePin(ROW3_GPIO_Port, ROW3_Pin, GPIO_PIN_RESET);
				
		current_row = 1;
		
		if (current_col != -1){
			if (HAL_GPIO_ReadPin(COL0_GPIO_Port, COL0_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 0 is 3 
				sprintf(pressed, "B");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL0_GPIO_Port, COL0_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL1_GPIO_Port, COL1_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 1 is 2 
				sprintf(pressed, "A");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL1_GPIO_Port, COL1_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL2_GPIO_Port, COL2_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "9");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL2_GPIO_Port, COL2_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL3_GPIO_Port, COL3_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "8");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL3_GPIO_Port, COL3_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
		}
		HAL_GPIO_WritePin(ROW0_GPIO_Port, ROW0_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW1_GPIO_Port, ROW1_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW2_GPIO_Port, ROW2_Pin, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(ROW3_GPIO_Port, ROW3_Pin, GPIO_PIN_SET);
		
		HAL_Delay(20);
		
		current_row = 1;
		
		if (current_col != -1){
			if (HAL_GPIO_ReadPin(COL0_GPIO_Port, COL0_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 0 is 3 
				sprintf(pressed, "F");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL0_GPIO_Port, COL0_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL1_GPIO_Port, COL1_Pin) == GPIO_PIN_SET)
			{
			//col 0 row 1 is 2 
				sprintf(pressed, "E");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL1_GPIO_Port, COL1_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL2_GPIO_Port, COL2_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "D");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL2_GPIO_Port, COL2_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
			if (HAL_GPIO_ReadPin(COL3_GPIO_Port, COL3_Pin) == GPIO_PIN_SET)
			{
				sprintf(pressed, "C");
				print_msg(pressed);
			
			//reset pin, and column
				HAL_GPIO_WritePin(COL3_GPIO_Port, COL3_Pin, GPIO_PIN_RESET);
				current_col = -1;
			}
		}
		//go back to top of loop (first row)
		
  }
}