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
  ******************************************************************************
  */
#include "main.h"
#include "config.h"
#include <stdio.h>


int8_t current_row = -1, current_col = -1;

typedef struct {
    GPIO_TypeDef *port;
    uint16_t pin;
} GPIO_Config;

// Define rows and cols arrays
GPIO_Config rows[4] = {
    {ROW0_GPIO_Port, ROW0_Pin},
    {ROW1_GPIO_Port, ROW1_Pin},
    {ROW2_GPIO_Port, ROW2_Pin},
    {ROW3_GPIO_Port, ROW3_Pin}
};

GPIO_Config cols[4] = {
    {COL0_GPIO_Port, COL0_Pin},
    {COL1_GPIO_Port, COL1_Pin},
    {COL2_GPIO_Port, COL2_Pin},
    {COL3_GPIO_Port, COL3_Pin}
};

int main(void)
{
  /* Reset of all peripherals. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  
  /* Initialize ROW outputs */
  HAL_GPIO_WritePin(ROW0_GPIO_Port, ROW0_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(ROW1_GPIO_Port, ROW1_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(ROW2_GPIO_Port, ROW2_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(ROW3_GPIO_Port, ROW3_Pin, GPIO_PIN_RESET);
	
	char space[2];
	sprintf(space, " ");
	print_msg(space);
	
	char pressed[5];
	
	
  char keypadValues[4][4] = {
    {'3', '2', '1', '0'},
    {'7', '6', '5', '4'},
    {'B', 'A', '9', '8'},
    {'F', 'E', 'D', 'C'}
	};



	int8_t key_pressed = 0;

	 /* Infinite loop */
	while (1)
	{
    HAL_Delay(150); // Small delay after LED toggle

    for (int row = 0; row < 4; ++row)
    {
        // Set one row to HIGH at a time
        for (int r = 0; r < 4; ++r)
        {
					//are u current row (set, else, reset)
            HAL_GPIO_WritePin(rows[r].port, rows[r].pin, (r == row) ? GPIO_PIN_SET : GPIO_PIN_RESET);
        }

        //small dleay 
        HAL_Delay(10);

        //col interrupts
        for (int col = 0; col < 4; ++col)
        {			
						//if col is set 
            if (HAL_GPIO_ReadPin(cols[col].port, cols[col].pin) == GPIO_PIN_SET)
            {
                //key pressed
                current_row = row;
                current_col = col;
							
								//reset the column
								HAL_GPIO_WritePin(cols[col].port, cols[col].pin, GPIO_PIN_RESET);
                
            }
        }
    }

    //print key
    if (current_row != -1 && current_col != -1)
    {
        //get value, print to UART
        char pressedValue = keypadValues[current_row][current_col];
        print_msg(&pressedValue); // Print the pressed value

        //reset current_col 
        current_col = -1;
    }
}
}